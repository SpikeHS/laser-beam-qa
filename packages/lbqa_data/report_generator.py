"""HTML/PDF report rendering for persisted laser beam QA runs."""

from __future__ import annotations

import base64
import csv
import html
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from lbqa_analysis.waist_refinement import minimum_area_from_zscan
from lbqa_contracts.errors import ErrorCode
from matplotlib.figure import Figure
from PIL import Image

from lbqa_data.beam_visualization import (
    BeamAxes,
    BeamDisplayOptions,
    beam_axes_from_measurement,
    crop_beam_region,
    estimate_beam_axes,
    render_beam_spot_image,
    save_beam_spot_png,
    stabilize_beam_axes,
)
from lbqa_data.data_exceptions import DataStoreError

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ModuleNotFoundError:
    Environment = None
    FileSystemLoader = None
    select_autoescape = None


class ReportGenerator:
    """Render a source-backed run report from persisted summary and z-scan data."""

    def __init__(self, template_dir: str | Path | None = None) -> None:
        self.template_dir = (
            Path(template_dir)
            if template_dir is not None
            else Path(__file__).with_name("report_templates")
        )
        if Environment is None or FileSystemLoader is None or select_autoescape is None:
            self._environment = None
        else:
            self._environment = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=select_autoescape(enabled_extensions=("html", "j2")),
            )

    def generate_html(
        self,
        *,
        run_dir: str | Path,
        summary: Mapping[str, Any],
        zscan_result: Any | None = None,
        plane_results: Sequence[Any] | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Render `report.html` for one run directory."""

        run_path = Path(run_dir)
        report_path = Path(output_path) if output_path is not None else run_path / "report.html"
        report_planes = list(plane_results) if plane_results is not None else _plane_results(
            zscan_result
        )
        fit_planes = _plane_results(zscan_result) or report_planes
        assets = _write_report_assets(run_path, report_planes, zscan_result, summary=summary)
        context = {
            "summary": summary,
            "sample_info": _mapping(summary.get("sample_info")),
            "recipe": _mapping(summary.get("recipe")),
            "calibration": _mapping(summary.get("calibration")),
            "system": _mapping(summary.get("system")),
            "morphology": _mapping(summary.get("morphology_summary")),
            "divergence": _mapping(summary.get("divergence_result")),
            "gaussian_prediction": _mapping(summary.get("gaussian_prediction")),
            "measured_beam_quality": _mapping(summary.get("measured_beam_quality")),
            "key_results": _mapping(summary.get("key_results")),
            "minimum_area": _mapping(summary.get("minimum_area")),
            "waist_refinement": _mapping(summary.get("waist_refinement")),
            "linear_model": _mapping(summary.get("divergence_result")).get("model")
            == "far_field_linear" or _is_linear(zscan_result),
            "errors": list(summary.get("error_list") or []),
            "invalid_points": _invalid_points(report_planes),
            "near_field_image_data_uri": _near_field_image_data_uri(run_path),
            "zscan_plot_data_uri": _zscan_plot_data_uri(fit_planes, zscan_result),
            "beam_propagation_plot_data_uri": _beam_propagation_plot_data_uri(
                fit_planes,
                zscan_result,
            ),
            "near_field_image_src": assets.get("near_field_image_src"),
            "zscan_plot_src": assets.get("zscan_plot_src"),
            "beam_propagation_plot_src": assets.get("beam_propagation_plot_src"),
            "beam_spot_images": list(assets.get("beam_spot_images") or []),
            "key_spot_images": list(assets.get("key_spot_images") or []),
            "fit_coefficients": _fit_coefficients(zscan_result),
            "output_files": _mapping(summary.get("output_files")),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if self._environment is None:
            rendered = _render_fallback_html(context)
        else:
            template = self._environment.get_template("default_report.html.j2")
            rendered = template.render(**context)
        report_path.write_text(rendered, encoding="utf-8")
        return report_path

    def generate_pdf(
        self,
        *,
        html_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path | None:
        """Optionally render PDF through WeasyPrint when the report extra is installed."""

        try:
            from weasyprint import HTML
        except ImportError:
            return None

        source = Path(html_path)
        pdf_path = Path(output_path) if output_path is not None else source.with_suffix(".pdf")
        HTML(filename=str(source)).write_pdf(str(pdf_path))
        return pdf_path

    def report_error(self, failure_payload: Mapping[str, Any]) -> None:
        """Compatibility hook for sequencer error recovery report services."""

        _ = failure_payload


def _plane_results(zscan_result: Any | None) -> list[Any]:
    if zscan_result is None:
        return []
    points = _read_field(zscan_result, "points", [])
    return list(points or [])


def _invalid_points(plane_results: Sequence[Any]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for index, point in enumerate(plane_results):
        if bool(_read_field(point, "valid", False)):
            continue
        invalid.append(
            {
                "index": index,
                "z_actual_mm": _read_field(point, "z_actual_mm", None),
                "invalid_reason": _read_field(point, "invalid_reason", ""),
            }
        )
    return invalid


def _write_report_assets(
    run_dir: Path,
    plane_results: Sequence[Any],
    zscan_result: Any | None,
    *,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    assets_dir = run_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Any] = {}
    spot_options = _display_options_from_summary(summary)
    image_entries = _scan_image_entries(run_dir, plane_results)
    shared_intensity_range = _shared_report_intensity_range(
        [np.asarray(entry["image"]) for entry in image_entries],
        spot_options,
    )
    spot_options = _options_with_intensity_range(spot_options, shared_intensity_range)
    tracked_axes: list[BeamAxes | None] = []
    previous_axes: BeamAxes | None = None
    for entry in image_entries:
        axes = _axes_for_report(
            np.asarray(entry["image"]),
            spot_options,
            point=entry.get("point"),
            summary=summary,
            metadata=entry.get("row"),
        )
        if axes is not None and axes.source != "scan_plane_continuous_axes":
            axes = stabilize_beam_axes(axes, previous_axes)
        if axes is not None:
            previous_axes = axes
        tracked_axes.append(axes)

    fit_planes = _plane_results(zscan_result) or list(plane_results)
    zscan_plot = _zscan_plot_image(fit_planes, zscan_result)
    if zscan_plot is not None:
        zscan_plot_path = assets_dir / "divergence_fit.png"
        zscan_plot.save(zscan_plot_path)
        assets["zscan_plot_src"] = zscan_plot_path.relative_to(run_dir).as_posix()

    propagation_plot = render_beam_propagation_image(fit_planes, zscan_result)
    if propagation_plot is not None:
        propagation_plot_path = assets_dir / "beam_propagation.png"
        propagation_plot.save(propagation_plot_path)
        assets["beam_propagation_plot_src"] = propagation_plot_path.relative_to(
            run_dir
        ).as_posix()

    spot_images = []
    for index, entry in enumerate(image_entries):
        point_index = entry["plane_index_count"]
        z_label = _format_z_label(entry.get("point"))
        spot_path = assets_dir / f"spot_{point_index:03d}.png"
        _save_spot_png(
            np.asarray(entry["image"]),
            spot_path,
            title=z_label,
            options=spot_options,
            point=entry.get("point"),
            axes=tracked_axes[index] if index < len(tracked_axes) else None,
        )
        caption = _image_caption(point_index, entry.get("point"), row=entry.get("row"))
        spot_images.append(
            {
                "src": spot_path.relative_to(run_dir).as_posix(),
                "caption": caption,
                "plane_index_count": point_index,
            }
        )
    assets["beam_spot_images"] = spot_images
    assets["key_spot_images"] = _key_spot_images(
        spot_images,
        _mapping(summary.get("key_results")),
    )
    if spot_images:
        assets["near_field_image_src"] = spot_images[0]["src"]
    return assets


def _near_field_image_data_uri(run_dir: Path) -> str | None:
    first_image = _first_scan_image(run_dir)
    if first_image is None:
        return None
    try:
        display_image = _normalize_image_for_png(first_image[1])
    except Exception as exc:
        raise DataStoreError(
            ErrorCode.E_REPORT_GENERATION_FAILED,
            f"failed to load report image {first_image[0]}: {exc}",
        ) from exc
    return _png_data_uri(display_image)


def _normalize_image_for_png(image: np.ndarray) -> Image.Image:
    return render_beam_spot_image(image, options=BeamDisplayOptions(overlay_mode="none"))


def _first_scan_image(run_dir: Path) -> tuple[Path, np.ndarray] | None:
    for path in _scan_image_paths(run_dir):
        try:
            return path, _load_scan_image(path)
        except Exception:
            continue
    return None


def _scan_image_entries(run_dir: Path, plane_results: Sequence[Any]) -> list[dict[str, Any]]:
    entries = []
    paths = _scan_image_paths(run_dir)
    scan_rows = _scan_table_rows(run_dir)
    for index, path in enumerate(paths):
        try:
            image = _load_scan_image(path)
        except Exception:
            continue
        frame_match = re.fullmatch(r"frame_(\d+)_corrected", path.stem)
        if frame_match is not None:
            index = int(frame_match.group(1))
        point = plane_results[index] if index < len(plane_results) else None
        row = scan_rows[index] if index < len(scan_rows) else None
        entries.append({"path": path, "image": image, "point": point, "row": row,
                        "plane_index_count": index})
    return entries


def _scan_table_rows(run_dir: Path) -> list[dict[str, str]]:
    csv_path = run_dir / "z_scan_table.csv"
    if not csv_path.is_file():
        return []
    try:
        with csv_path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    except OSError:
        return []


def _scan_image_paths(run_dir: Path) -> list[Path]:
    working_paths = sorted((run_dir / ".working").glob("*_corrected.npy"))
    if working_paths:
        return working_paths
    processed_paths = sorted((run_dir / "processed").glob("*_corrected.npy"))
    if processed_paths:
        return processed_paths
    raw_paths = sorted((run_dir / "images").glob("*_raw.tiff"))
    raw_paths.extend(sorted((run_dir / "images").glob("*_raw.tif")))
    raw_paths.extend(sorted((run_dir / "images").glob("dark_frame*.tiff")))
    return raw_paths


def _load_scan_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    with Image.open(path) as image:
        return np.asarray(image)


def _save_normalized_png(
    image: np.ndarray,
    output_path: Path,
    *,
    options: BeamDisplayOptions,
    point: Any | None = None,
    axes: BeamAxes | None = None,
) -> None:
    report_options = _report_options_for_image(image, options, axes=axes)
    save_beam_spot_png(
        image,
        output_path,
        title="near field",
        options=report_options,
        rotation_deg=_report_rotation_deg(point, axes=axes),
        axes=axes or _axes_for_report(image, report_options),
    )


def _save_spot_png(
    image: np.ndarray,
    output_path: Path,
    *,
    title: str,
    options: BeamDisplayOptions,
    point: Any | None,
    axes: BeamAxes | None = None,
) -> None:
    report_options = _report_options_for_image(image, options, axes=axes)
    if _continuous_axes(point):
        x_um = _optional_float(_read_field(point, "d4sigma_x_um", None))
        y_um = _optional_float(_read_field(point, "d4sigma_y_um", None))
        if x_um is not None and y_um is not None:
            title = f"{title} | X={x_um:.3g} Y={y_um:.3g} um"
    save_beam_spot_png(
        image,
        output_path,
        title=title,
        options=report_options,
        rotation_deg=_report_rotation_deg(point, axes=axes),
        axes=axes or _axes_for_report(image, report_options),
    )


def _save_surface_png(
    image: np.ndarray,
    output_path: Path,
    *,
    title: str,
    options: BeamDisplayOptions,
    intensity_range: tuple[float, float] | None = None,
) -> None:
    report_options = _report_options_for_image(image, options)
    crop = crop_beam_region(image, options=report_options)
    display = _downsample_for_plot(crop.image, max_size_px=96)
    if intensity_range is not None:
        display = _scale_surface_for_shared_range(display, intensity_range)
    y_values = np.arange(display.shape[0])
    x_values = np.arange(display.shape[1])
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    figure = Figure(figsize=(3.4, 3.0), dpi=120)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        x_grid,
        y_grid,
        display.astype(np.float64, copy=False),
        cmap="viridis",
        vmin=0.0 if intensity_range is not None else None,
        vmax=1.0 if intensity_range is not None else None,
        linewidth=0,
        antialiased=False,
    )
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    if intensity_range is not None:
        axis.set_zlim(0.0, 1.0)
    axis.view_init(elev=34, azim=-58)
    figure.tight_layout(pad=0.3)
    _save_figure_png(figure, output_path)


def _display_options_from_summary(
    summary: Mapping[str, Any],
    *,
    max_display_px: int = 640,
) -> BeamDisplayOptions:
    system = _mapping(summary.get("system"))
    visualization = _mapping(system.get("visualization"))
    linear = _mapping(summary.get("divergence_result")).get("model") == "far_field_linear"
    return BeamDisplayOptions(
        threshold_percent=float(visualization.get("range_threshold_percent", 50.0)),
        max_display_px=max_display_px,
        overlay_mode="box" if linear else str(visualization.get("overlay_mode", "box")),
        colormap="thermal" if linear else str(visualization.get("colormap", "thermal")),
        zoom_factor=float(visualization.get("zoom_factor", 1.0)),
        center_x_px=_optional_float(visualization.get("center_x_px")),
        center_y_px=_optional_float(visualization.get("center_y_px")),
    )


def _report_options_for_image(
    image: np.ndarray,
    options: BeamDisplayOptions,
    *,
    axes: BeamAxes | None = None,
) -> BeamDisplayOptions:
    """Auto-crop report beam images when the operator did not choose a crop."""

    if (
        options.zoom_factor > 1.0
        or options.center_x_px is not None
        or options.center_y_px is not None
    ):
        return options

    axes = axes or _axes_for_report(image, options)
    if axes is None:
        return options
    zoom_factor = _auto_report_zoom_factor(image, axes)
    if zoom_factor <= 1.0:
        return options
    return BeamDisplayOptions(
        threshold_percent=options.threshold_percent,
        min_display_px=options.min_display_px,
        max_display_px=options.max_display_px,
        overlay_mode=options.overlay_mode,
        colormap=options.colormap,
        zoom_factor=zoom_factor,
        center_x_px=axes.centroid_x_px,
        center_y_px=axes.centroid_y_px,
        intensity_min=options.intensity_min,
        intensity_max=options.intensity_max,
    )


def _shared_report_intensity_range(
    images: Sequence[np.ndarray],
    options: BeamDisplayOptions,
) -> tuple[float, float] | None:
    values: list[tuple[float, float]] = []
    for image in images:
        try:
            report_options = _report_options_for_image(image, options)
            crop = crop_beam_region(image, options=report_options)
            values.append((float(np.min(crop.image)), float(np.max(crop.image))))
        except Exception:
            continue
    if not values:
        return None
    min_value = min(item[0] for item in values)
    max_value = max(item[1] for item in values)
    if max_value <= min_value:
        return None
    return min_value, max_value


def _options_with_intensity_range(
    options: BeamDisplayOptions,
    intensity_range: tuple[float, float] | None,
) -> BeamDisplayOptions:
    if intensity_range is None:
        return options
    return BeamDisplayOptions(
        threshold_percent=options.threshold_percent,
        min_display_px=options.min_display_px,
        max_display_px=options.max_display_px,
        overlay_mode=options.overlay_mode,
        colormap=options.colormap,
        zoom_factor=options.zoom_factor,
        center_x_px=options.center_x_px,
        center_y_px=options.center_y_px,
        intensity_min=float(intensity_range[0]),
        intensity_max=float(intensity_range[1]),
    )


def _scale_surface_for_shared_range(
    image: np.ndarray,
    intensity_range: tuple[float, float],
) -> np.ndarray:
    min_value, max_value = intensity_range
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float64)
    return np.clip((image.astype(np.float64) - min_value) / (max_value - min_value), 0.0, 1.0)


def _axes_for_report(
    image: np.ndarray,
    options: BeamDisplayOptions,
    *,
    point: Any | None = None,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BeamAxes | None:
    point_axes = _axes_from_plane_result(point, summary or {}, options, metadata=metadata)
    if point_axes is not None:
        return point_axes
    return estimate_beam_axes(image, threshold_percent=options.threshold_percent)


def _axes_from_plane_result(
    point: Any | None,
    summary: Mapping[str, Any],
    options: BeamDisplayOptions,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> BeamAxes | None:
    if point is None:
        return None
    system = _mapping(summary.get("system"))
    pixel_x_um = _optional_float(system.get("effective_pixel_x_um"))
    pixel_y_um = _optional_float(system.get("effective_pixel_y_um"))
    if pixel_x_um is None or pixel_y_um is None or pixel_x_um <= 0.0 or pixel_y_um <= 0.0:
        return None
    continuous = _continuous_axes(point) or _mapping(summary.get("divergence_result")).get(
        "model"
    ) == "far_field_linear"
    try:
        centroid_x_um = float(_read_field(point, "centroid_x_um", None))
        centroid_y_um = float(_read_field(point, "centroid_y_um", None))
        major_um = float(_read_field(point, "d4sigma_major_um", None))
        minor_um = float(_read_field(point, "d4sigma_minor_um", None))
        azimuth_deg = float(_read_field(point, "azimuth_deg", None))
    except (TypeError, ValueError):
        return None
    values = (centroid_x_um, centroid_y_um, major_um, minor_um, azimuth_deg)
    if not all(math.isfinite(value) for value in values) or major_um <= 0.0 or minor_um <= 0.0:
        return None
    mean_pixel_um = (pixel_x_um + pixel_y_um) / 2.0
    if continuous:
        x_um = _optional_float(_read_field(point, "d4sigma_x_um", None))
        y_um = _optional_float(_read_field(point, "d4sigma_y_um", None))
        if x_um is None or y_um is None or x_um <= 0.0 or y_um <= 0.0:
            return None
        metadata = metadata or {}
        x_angle = _optional_float(metadata.get("axis_x_azimuth_deg"))
        if x_angle is None:
            x_angle = azimuth_deg
        native_angle = _optional_float(metadata.get("native_major_azimuth_deg"))
        uncertain = str(metadata.get("axis_orientation_uncertain", False)).lower() == "true"
        if native_angle is None:
            if uncertain:
                return None
            native_angle = x_angle if x_um >= y_um else x_angle + 90.0
        axes = beam_axes_from_measurement(
            centroid_x_um=centroid_x_um,
            centroid_y_um=centroid_y_um,
            major_diameter_um=major_um,
            minor_diameter_um=minor_um,
            major_azimuth_deg=native_angle,
            x_diameter_um=x_um,
            y_diameter_um=y_um,
            x_azimuth_deg=x_angle,
            pixel_x_um=pixel_x_um,
            pixel_y_um=pixel_y_um,
            threshold_percent=options.threshold_percent,
            source="scan_plane_continuous_axes",
        )
        return replace(
            axes,
            axis_assignment="major_to_x" if x_um >= y_um else "minor_to_x",
            orientation_status="held_near_circular" if uncertain else "continuous",
        )
    return BeamAxes(
        centroid_x_px=centroid_x_um / pixel_x_um,
        centroid_y_px=centroid_y_um / pixel_y_um,
        azimuth_deg=azimuth_deg,
        major_sigma_px=major_um / (4.0 * mean_pixel_um),
        minor_sigma_px=minor_um / (4.0 * mean_pixel_um),
        threshold_percent=options.threshold_percent,
        source="scan_plane_gaussian_fit",
    )


def _auto_report_zoom_factor(image: np.ndarray, axes: BeamAxes) -> float:
    frame = np.asarray(image)
    if frame.ndim != 2:
        frame = np.squeeze(frame)
    if frame.ndim != 2:
        return 1.0
    height_px, width_px = frame.shape
    target_span_px = max(
        80.0,
        float(axes.major_sigma_px) * 8.0,
        float(axes.minor_sigma_px) * 8.0,
    )
    target_span_px = min(target_span_px, float(max(width_px, height_px)))
    if target_span_px <= 0.0:
        return 1.0
    return min(24.0, max(1.0, min(width_px, height_px) / target_span_px))


def _save_figure_png(figure: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png", bbox_inches="tight")


def _downsample_for_plot(image: np.ndarray, *, max_size_px: int) -> np.ndarray:
    image_float = np.asarray(image, dtype=np.float32)
    if image_float.ndim != 2:
        image_float = np.squeeze(image_float)
    if image_float.ndim != 2:
        raise ValueError(f"report image must be 2D, got shape {image_float.shape}.")
    height_px, width_px = image_float.shape
    longest = max(height_px, width_px)
    if longest <= max_size_px:
        return image_float
    stride = max(1, int(math.ceil(longest / float(max_size_px))))
    return image_float[::stride, ::stride]


def _format_z_label(point: Any) -> str:
    z_actual_mm = _read_field(point, "z_actual_mm", None)
    if z_actual_mm is None:
        return "z: --"
    return f"z={float(z_actual_mm):.6f} mm"


def _image_caption(
    index: int,
    point: Any,
    *,
    row: Mapping[str, Any] | None = None,
) -> str:
    z_actual_mm = _read_field(point, "z_actual_mm", None)
    d4sigma_x_um = _read_field(point, "d4sigma_x_um", None)
    d4sigma_y_um = _read_field(point, "d4sigma_y_um", None)
    azimuth_deg = _read_field(point, "azimuth_deg", None)
    valid = _read_field(point, "valid", None)
    parts = [f"Point {index + 1}"]
    if row and row.get("scan_phase"):
        parts.append(f"phase={row['scan_phase']}")
    if z_actual_mm is not None:
        parts.append(f"z={float(z_actual_mm):.6f} mm")
    if d4sigma_x_um is not None and d4sigma_y_um is not None:
        parts.append(f"width X/Y={float(d4sigma_x_um):.3g}/{float(d4sigma_y_um):.3g} um")
    if azimuth_deg is not None:
        label = "X-axis angle" if _continuous_axes(point) else "azimuth"
        parts.append(f"{label}={float(azimuth_deg):.2f} deg")
    if row and "fit_included_x" in row:
        parts.append(f"fit X/Y={row['fit_included_x']}/{row.get('fit_included_y', '')}")
    if valid is not None:
        parts.append(f"valid={bool(valid)}")
    return " | ".join(parts)


def _key_spot_images(
    spot_images: Sequence[Mapping[str, Any]],
    key_results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selections = (
        ("X minimum", _mapping(key_results.get("x")).get("image_plane_index_count")),
        ("Y minimum", _mapping(key_results.get("y")).get("image_plane_index_count")),
        (
            "Minimum-area plane",
            _mapping(key_results.get("minimum_area")).get("nearest_plane_index_count"),
        ),
    )
    selected = []
    for role, index_value in selections:
        try:
            index = int(index_value)
        except (TypeError, ValueError):
            continue
        matching = next(
            (item for offset, item in enumerate(spot_images)
             if item.get("plane_index_count", offset) == index), None
        )
        if matching is None:
            continue
        item = dict(matching)
        item["role"] = role
        item["plane_index_count"] = index
        selected.append(item)
    return selected


def _report_rotation_deg(
    point: Any | None,
    *,
    axes: BeamAxes | None = None,
) -> float | None:
    if axes is not None and axes.x_axis_azimuth_deg is not None:
        return -float(axes.x_axis_azimuth_deg)
    azimuth_deg = _read_field(point, "azimuth_deg", None)
    if azimuth_deg is None:
        return None
    try:
        azimuth = float(azimuth_deg)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(azimuth):
        return None
    return -azimuth


def _zscan_plot_data_uri(plane_results: Sequence[Any], zscan_result: Any | None) -> str | None:
    image = _zscan_plot_image(plane_results, zscan_result)
    if image is None:
        return None
    return _png_data_uri(image)


def _beam_propagation_plot_data_uri(
    plane_results: Sequence[Any],
    zscan_result: Any | None,
) -> str | None:
    image = render_beam_propagation_image(plane_results, zscan_result)
    if image is None:
        return None
    return _png_data_uri(image)


def _zscan_plot_image(plane_results: Sequence[Any], zscan_result: Any | None) -> Image.Image | None:
    if not plane_results:
        return None

    z_actual_mm = np.asarray(
        [float(_read_field(point, "z_actual_mm", 0.0)) for point in plane_results],
        dtype=np.float64,
    )
    d4sigma_x_um = np.asarray(
        [float(_read_field(point, "d4sigma_x_um", 0.0)) for point in plane_results],
        dtype=np.float64,
    )
    d4sigma_y_um = np.asarray(
        [float(_read_field(point, "d4sigma_y_um", 0.0)) for point in plane_results],
        dtype=np.float64,
    )
    quality_invalid_mask = np.asarray(
        [not bool(_read_field(point, "valid", False)) for point in plane_results],
        dtype=bool,
    )
    figure = Figure(figsize=(7.5, 4.2), dpi=140)
    axis = figure.subplots()
    for name, diameter, color, marker in (
        ("x", d4sigma_x_um, "#2563eb", "o"),
        ("y", d4sigma_y_um, "#dc2626", "s"),
    ):
        selected = _fit_inlier_mask(plane_results, zscan_result, axis_name=name)
        excluded = ~(selected | quality_invalid_mask)
        axis.scatter(
            z_actual_mm[selected], diameter[selected],
            label=f"Width {name.upper()} included", color=color, marker=marker, s=28,
        )
        if np.any(excluded):
            axis.scatter(
                z_actual_mm[excluded], diameter[excluded],
                label=f"Width {name.upper()} excluded", facecolors="none",
                edgecolors="#6b7280", marker=marker, s=42,
            )
        if np.any(quality_invalid_mask):
            axis.scatter(
                z_actual_mm[quality_invalid_mask], diameter[quality_invalid_mask],
                label=f"Quality invalid {name.upper()}", color="#6b7280", marker="x", s=42,
            )

    _plot_fit(axis, z_actual_mm, zscan_result, axis_name="x", color="#1d4ed8")
    _plot_fit(axis, z_actual_mm, zscan_result, axis_name="y", color="#b91c1c")
    _plot_waist_markers(axis, zscan_result, plane_results)

    axis.set_xlabel("z_actual_mm")
    axis.set_ylabel("Beam width diameter (um)")
    axis.grid(True, color="#e5e7eb", linewidth=0.8)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()

    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB").copy()


def _plot_waist_markers(
    axis: Any,
    zscan_result: Any | None,
    plane_results: Sequence[Any],
) -> None:
    if zscan_result is None or _is_linear(zscan_result):
        return
    markers = (
        ("waist_z_x_mm", "X waist", "#1d4ed8", "--"),
        ("waist_z_y_mm", "Y waist", "#b91c1c", "--"),
    )
    for field_name, label, color, line_style in markers:
        z_mm = _read_field(zscan_result, field_name, None)
        if z_mm is not None:
            axis.axvline(
                float(z_mm),
                color=color,
                linestyle=line_style,
                linewidth=1.0,
                alpha=0.65,
                label=label,
            )
    minimum_area = minimum_area_from_zscan(zscan_result, plane_results=plane_results)
    if minimum_area.get("valid"):
        axis.axvline(
            float(minimum_area["z_mm"]),
            color="#15803d",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label="Minimum area",
        )


def render_beam_propagation_image(
    plane_results: Sequence[Any],
    zscan_result: Any | None,
) -> Image.Image | None:
    """Render centroid drift and fitted X/Y beam envelopes versus z."""

    if not plane_results:
        return None

    z_actual_mm = np.asarray(
        [float(_read_field(point, "z_actual_mm", 0.0)) for point in plane_results],
        dtype=np.float64,
    )
    centroid_x_um = np.asarray(
        [float(_read_field(point, "centroid_x_um", 0.0)) for point in plane_results],
        dtype=np.float64,
    )
    centroid_y_um = np.asarray(
        [float(_read_field(point, "centroid_y_um", 0.0)) for point in plane_results],
        dtype=np.float64,
    )
    width_x_um = np.asarray(
        [float(_read_field(point, "d4sigma_x_um", 0.0)) for point in plane_results],
        dtype=np.float64,
    )
    width_y_um = np.asarray(
        [float(_read_field(point, "d4sigma_y_um", 0.0)) for point in plane_results],
        dtype=np.float64,
    )
    selected_x = _fit_inlier_mask(plane_results, zscan_result, axis_name="x")
    selected_y = _fit_inlier_mask(plane_results, zscan_result, axis_name="y")
    quality_invalid_mask = np.asarray(
        [not bool(_read_field(point, "valid", False)) for point in plane_results],
        dtype=bool,
    )

    figure = Figure(figsize=(8.6, 8.0), dpi=140)
    axes = figure.subplots(3, 1, sharex=True)
    centroid_axis, x_axis, y_axis = axes
    _plot_centroid_drift(
        centroid_axis,
        z_actual_mm,
        centroid_x_um,
        centroid_y_um,
        selected_x,
        ~(selected_x | quality_invalid_mask),
        selected_mask_y=selected_y,
        excluded_mask_y=~(selected_y | quality_invalid_mask),
    )
    _plot_beam_envelope(
        x_axis,
        z_actual_mm=z_actual_mm,
        centroid_um=centroid_x_um,
        width_um=width_x_um,
        selected_mask=selected_x,
        excluded_mask=~(selected_x | quality_invalid_mask),
        zscan_result=zscan_result,
        axis_name="x",
        color="#2563eb",
        title="X-axis beam envelope",
    )
    _plot_beam_envelope(
        y_axis,
        z_actual_mm=z_actual_mm,
        centroid_um=centroid_y_um,
        width_um=width_y_um,
        selected_mask=selected_y,
        excluded_mask=~(selected_y | quality_invalid_mask),
        zscan_result=zscan_result,
        axis_name="y",
        color="#c2410c",
        title="Y-axis beam envelope",
    )
    y_axis.set_xlabel("z position (mm)")
    figure.suptitle(
        "Beam centroid and far-field linear propagation" if _is_linear(zscan_result)
        else "Beam centroid and Gaussian-fit propagation",
        fontsize=12,
        color="#172033",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB").copy()


def _plot_centroid_drift(
    axis: Any,
    z_actual_mm: np.ndarray,
    centroid_x_um: np.ndarray,
    centroid_y_um: np.ndarray,
    selected_mask: np.ndarray,
    excluded_mask: np.ndarray,
    *,
    selected_mask_y: np.ndarray | None = None,
    excluded_mask_y: np.ndarray | None = None,
) -> None:
    selected_mask_y = selected_mask if selected_mask_y is None else selected_mask_y
    excluded_mask_y = excluded_mask if excluded_mask_y is None else excluded_mask_y
    axis.plot(
        z_actual_mm[selected_mask],
        centroid_x_um[selected_mask],
        color="#2563eb",
        marker="o",
        linewidth=1.4,
        label="Centroid X",
    )
    axis.plot(
        z_actual_mm[selected_mask_y],
        centroid_y_um[selected_mask_y],
        color="#c2410c",
        marker="s",
        linestyle="--",
        linewidth=1.4,
        label="Centroid Y",
    )
    if np.any(excluded_mask):
        axis.scatter(
            z_actual_mm[excluded_mask],
            centroid_x_um[excluded_mask],
            facecolors="none",
            edgecolors="#6b7280",
            marker="o",
            s=42,
            label="Excluded",
        )
    if np.any(excluded_mask_y):
        axis.scatter(
            z_actual_mm[excluded_mask_y],
            centroid_y_um[excluded_mask_y],
            color="#6b7280",
            marker="x",
            s=38,
        )
    axis.set_ylabel("Centroid (um)")
    axis.set_title("Beam centroid drift", loc="left", fontsize=10)
    axis.grid(True, color="#e5e7eb", linewidth=0.8)
    axis.legend(loc="best", fontsize=8, ncols=3)


def _plot_beam_envelope(
    axis: Any,
    *,
    z_actual_mm: np.ndarray,
    centroid_um: np.ndarray,
    width_um: np.ndarray,
    selected_mask: np.ndarray,
    excluded_mask: np.ndarray,
    zscan_result: Any | None,
    axis_name: str,
    color: str,
    title: str,
) -> None:
    if np.any(selected_mask):
        axis.errorbar(
            z_actual_mm[selected_mask],
            centroid_um[selected_mask],
            yerr=width_um[selected_mask] / 2.0,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=3,
            markersize=4,
            label="Included measurement",
        )
    if np.any(excluded_mask):
        axis.errorbar(
            z_actual_mm[excluded_mask],
            centroid_um[excluded_mask],
            yerr=width_um[excluded_mask] / 2.0,
            fmt="x",
            color="#6b7280",
            ecolor="#9ca3af",
            elinewidth=0.8,
            capsize=2,
            markersize=5,
            label="Excluded measurement",
        )

    if len(np.unique(z_actual_mm[selected_mask])) >= 2:
        fit_z = z_actual_mm[selected_mask] if _is_linear(zscan_result) else z_actual_mm
        z_min = float(np.min(fit_z))
        z_max = float(np.max(fit_z))
        z_line = np.linspace(z_min, z_max, 160)
        centroid_line = _linear_centroid_fit(
            z_actual_mm[selected_mask],
            centroid_um[selected_mask],
            z_line,
        )
        diameter_line = _axis_diameter_curve(z_line, zscan_result, axis_name)
        if diameter_line is not None:
            lower = centroid_line - diameter_line / 2.0
            upper = centroid_line + diameter_line / 2.0
            axis.fill_between(
                z_line,
                lower,
                upper,
                color=color,
                alpha=0.14,
                label="Fitted beam envelope",
            )
            axis.plot(z_line, lower, color=color, linewidth=1.2)
            axis.plot(z_line, upper, color=color, linewidth=1.2)
            axis.plot(
                z_line,
                centroid_line,
                color="#374151",
                linestyle="--",
                linewidth=1.0,
                label="Centroid trend",
            )

    axis.set_ylabel(f"{axis_name.upper()} position (um)")
    axis.set_title(title, loc="left", fontsize=10)
    axis.grid(True, color="#e5e7eb", linewidth=0.8)
    axis.legend(loc="best", fontsize=8, ncols=2)


def _linear_centroid_fit(
    z_selected_mm: np.ndarray,
    centroid_selected_um: np.ndarray,
    z_line_mm: np.ndarray,
) -> np.ndarray:
    coefficients = np.polyfit(z_selected_mm, centroid_selected_um, deg=1)
    return np.polyval(coefficients, z_line_mm)


def _axis_diameter_curve(
    z_line_mm: np.ndarray,
    zscan_result: Any | None,
    axis_name: str,
) -> np.ndarray | None:
    if zscan_result is None:
        return None
    if _is_linear(zscan_result):
        fit = _mapping(_read_field(zscan_result, "fit_diagnostics", {})).get(axis_name, {})
        slope = _optional_float(fit.get("slope_um_per_mm"))
        reference = _optional_float(fit.get("z_reference_mm"))
        at_reference = _optional_float(fit.get("intercept_at_reference_um"))
        if slope is None:
            return None
        if reference is not None and at_reference is not None:
            return at_reference + slope * (z_line_mm - reference)
        intercept = _optional_float(fit.get("intercept_um"))
        return None if intercept is None else intercept + slope * z_line_mm
    full_angle_mrad = _read_field(zscan_result, f"full_angle_{axis_name}_mrad", None)
    waist_z_mm = _read_field(zscan_result, f"waist_z_{axis_name}_mm", None)
    waist_diameter_um = _read_field(zscan_result, f"waist_diameter_{axis_name}_um", None)
    if None in {full_angle_mrad, waist_z_mm, waist_diameter_um}:
        return None
    return np.sqrt(
        float(waist_diameter_um) ** 2
        + (float(full_angle_mrad) * (z_line_mm - float(waist_z_mm))) ** 2
    )


def _fit_inlier_mask(
    plane_results: Sequence[Any],
    zscan_result: Any | None,
    *,
    axis_name: str | None = None,
) -> np.ndarray:
    if zscan_result is None:
        return np.asarray(
            [bool(_read_field(point, "valid", False)) for point in plane_results],
            dtype=bool,
        )
    fit_points = list(_read_field(zscan_result, "points", []) or [])
    outlier_indices = set(_read_field(zscan_result, "outlier_indices", []) or [])
    fit_diagnostics = _read_field(zscan_result, "fit_diagnostics", {})
    if _is_linear(zscan_result):
        selected = np.zeros(len(plane_results), dtype=bool)
        diagnostics = {
            int(point["index"]): point for point in fit_diagnostics.get("points", [])
            if isinstance(point, Mapping) and "index" in point
        }
        axes = (axis_name,) if axis_name is not None else ("x", "y")
        for fit_index, plane_index in enumerate(
            _match_fit_points_to_plane_results(fit_points, plane_results)
        ):
            if plane_index is None:
                continue
            diagnostic = diagnostics.get(fit_index, {})
            selected[plane_index] = any(
                bool(diagnostic.get(
                    f"fit_included_{name}",
                    fit_index in _mapping(fit_diagnostics.get(name)).get("fit_indices", []),
                )) for name in axes
            )
        return selected
    diagnostic_included = {}
    if isinstance(fit_diagnostics, Mapping):
        for diagnostic in fit_diagnostics.get("points", []) or []:
            if isinstance(diagnostic, Mapping) and "index" in diagnostic:
                diagnostic_included[int(diagnostic["index"])] = bool(
                    diagnostic.get("fit_included", False)
                )
    selected = np.zeros(len(plane_results), dtype=bool)
    for fit_index, plane_index in enumerate(
        _match_fit_points_to_plane_results(fit_points, plane_results)
    ):
        if plane_index is None:
            continue
        point = fit_points[fit_index]
        default_included = (
            bool(_read_field(point, "valid", False))
            and not bool(_read_field(point, "outlier", False))
            and fit_index not in outlier_indices
        )
        selected[plane_index] = diagnostic_included.get(fit_index, default_included)
    return selected


def _match_fit_points_to_plane_results(
    fit_points: Sequence[Any],
    plane_results: Sequence[Any],
) -> list[int | None]:
    available = set(range(len(plane_results)))
    matched = []
    for fit_point in fit_points:
        match = next(
            (
                index
                for index in reversed(range(len(plane_results)))
                if index in available and plane_results[index] is fit_point
            ),
            None,
        )
        if match is None:
            match = next(
                (
                    index
                    for index in reversed(range(len(plane_results)))
                    if index in available
                    and _same_plane_measurement(fit_point, plane_results[index])
                ),
                None,
            )
        matched.append(match)
        if match is not None:
            available.remove(match)
    return matched


def _same_plane_measurement(left: Any, right: Any) -> bool:
    for field_name in (
        "z_actual_mm",
        "centroid_x_um",
        "centroid_y_um",
        "d4sigma_x_um",
        "d4sigma_y_um",
    ):
        try:
            left_value = float(_read_field(left, field_name, math.nan))
            right_value = float(_read_field(right, field_name, math.nan))
        except (TypeError, ValueError):
            return False
        if not math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12):
            return False
    return True


def _plot_fit(
    axis: Any,
    z_actual_mm: np.ndarray,
    zscan_result: Any,
    *,
    axis_name: str,
    color: str,
) -> None:
    if zscan_result is None or len(z_actual_mm) < 2:
        return
    if _is_linear(zscan_result):
        points = _plane_results(zscan_result)
        selected = _fit_inlier_mask(points, zscan_result, axis_name=axis_name)
        selected_z = np.asarray([
            float(_read_field(point, "z_actual_mm", math.nan))
            for point, included in zip(points, selected, strict=True) if included
        ])
        selected_z = selected_z[np.isfinite(selected_z)]
        if len(np.unique(selected_z)) < 2:
            return
        z_line = np.linspace(float(np.min(selected_z)), float(np.max(selected_z)), 120)
        diameter_line = _axis_diameter_curve(z_line, zscan_result, axis_name)
        if diameter_line is not None:
            axis.plot(z_line, diameter_line, color=color, linewidth=1.4, alpha=0.85)
        return
    full_angle_mrad = _read_field(zscan_result, f"full_angle_{axis_name}_mrad", None)
    waist_z_mm = _read_field(zscan_result, f"waist_z_{axis_name}_mm", None)
    waist_diameter_um = _read_field(zscan_result, f"waist_diameter_{axis_name}_um", None)
    if None in {full_angle_mrad, waist_z_mm, waist_diameter_um}:
        return
    z_min = float(np.min(z_actual_mm))
    z_max = float(np.max(z_actual_mm))
    z_line = np.linspace(z_min, z_max, 120)
    diameter_line = np.sqrt(
        float(waist_diameter_um) ** 2
        + (float(full_angle_mrad) * (z_line - float(waist_z_mm))) ** 2
    )
    axis.plot(z_line, diameter_line, color=color, linewidth=1.4, alpha=0.85)


def _fit_coefficients(zscan_result: Any | None) -> dict[str, dict[str, float | None]]:
    if zscan_result is None:
        return {"x": _empty_fit(), "y": _empty_fit()}
    return {
        "x": _axis_fit_coefficients(zscan_result, "x"),
        "y": _axis_fit_coefficients(zscan_result, "y"),
    }


def _axis_fit_coefficients(zscan_result: Any, axis_name: str) -> dict[str, float | None]:
    if _is_linear(zscan_result):
        return _empty_fit()
    full_angle_mrad = _read_field(zscan_result, f"full_angle_{axis_name}_mrad", None)
    waist_z_mm = _read_field(zscan_result, f"waist_z_{axis_name}_mm", None)
    waist_diameter_um = _read_field(zscan_result, f"waist_diameter_{axis_name}_um", None)
    if None in {full_angle_mrad, waist_z_mm, waist_diameter_um}:
        return _empty_fit()
    fit_a_um2_per_mm2 = float(full_angle_mrad) ** 2
    fit_b_um2_per_mm = -2.0 * fit_a_um2_per_mm2 * float(waist_z_mm)
    fit_c_um2 = float(waist_diameter_um) ** 2 + fit_a_um2_per_mm2 * float(waist_z_mm) ** 2
    return {
        "fit_a_um2_per_mm2": fit_a_um2_per_mm2,
        "fit_b_um2_per_mm": fit_b_um2_per_mm,
        "fit_c_um2": fit_c_um2,
    }


def _empty_fit() -> dict[str, float | None]:
    return {
        "fit_a_um2_per_mm2": None,
        "fit_b_um2_per_mm": None,
        "fit_c_um2": None,
    }


def _png_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _read_field(source: Any, field_name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(field_name, default)
    value = getattr(source, field_name, default)
    if isinstance(value, float) and not math.isfinite(value):
        return default
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_linear(zscan_result: Any) -> bool:
    return _mapping(_read_field(zscan_result, "fit_diagnostics", {})).get(
        "model"
    ) == "far_field_linear"


def _continuous_axes(point: Any) -> bool:
    return "continuous_axes" in str(_read_field(point, "width_source", ""))


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _render_fallback_html(context: Mapping[str, Any]) -> str:
    """Dependency-light report used when Jinja2 is unavailable in source checkout."""

    sample_info = _mapping(context.get("sample_info"))
    recipe = _mapping(context.get("recipe"))
    divergence = _mapping(context.get("divergence"))
    gaussian_prediction = _mapping(context.get("gaussian_prediction"))
    morphology = _mapping(context.get("morphology"))
    system = _mapping(context.get("system"))
    output_files = _mapping(context.get("output_files"))
    errors = list(context.get("errors") or [])
    near_field_image_data_uri = context.get("near_field_image_src") or context.get(
        "near_field_image_data_uri"
    )
    zscan_plot_data_uri = context.get("zscan_plot_data_uri")
    beam_propagation_plot_data_uri = context.get("beam_propagation_plot_data_uri")
    key_results = _mapping(context.get("key_results"))
    key_x = _mapping(key_results.get("x"))
    key_y = _mapping(key_results.get("y"))
    key_area = _mapping(key_results.get("minimum_area"))
    beam_spot_images = list(context.get("beam_spot_images") or [])
    final_judgement = _mapping(context.get("summary")).get("final_judgement", "UNKNOWN")
    linear = bool(context.get("linear_model"))
    summary = _mapping(context.get("summary"))
    measurement = summary.get("measurement_status", "in_progress")
    model_note = (
        "Far-field OLS: D(z)=intercept_at_reference+slope*(z-z_reference). "
        "Independent X/Y selections; gray points are excluded. No waist is inferred. "
        "Minima and ellipse area are observed across all valid main and fine planes. "
        "Unconfirmed minima provide candidate estimates only; two points have no residual std."
        if linear else "D(z)=sqrt(D0^2+Theta^2(z-z0)^2)"
    )
    divergence_rows = _table_rows(
        (
            ("full-angle divergence X (mrad)", divergence.get("full_angle_x_mrad")),
            ("full-angle divergence Y (mrad)", divergence.get("full_angle_y_mrad")),
            ("half-angle divergence X (mrad)", divergence.get("half_angle_x_mrad")),
            ("half-angle divergence Y (mrad)", divergence.get("half_angle_y_mrad")),
            ("fit R2 X", divergence.get("fit_r2_x")),
            ("fit R2 Y", divergence.get("fit_r2_y")),
        )
    )
    morphology_rows = _table_rows(
        (
            ("Valid points", morphology.get("valid_points_count")),
            ("Width X mean (um)", morphology.get("d4sigma_x_mean_um")),
            ("Width Y mean (um)", morphology.get("d4sigma_y_mean_um")),
            ("Ellipticity mean", morphology.get("ellipticity_mean")),
        )
    )
    prediction_rows = _table_rows(
        (
            ("Wavelength (nm)", gaussian_prediction.get("wavelength_nm")),
            ("M2", gaussian_prediction.get("m2")),
            (
                "Predicted full-angle X (mrad)",
                gaussian_prediction.get("predicted_full_angle_x_mrad"),
            ),
            (
                "Predicted full-angle Y (mrad)",
                gaussian_prediction.get("predicted_full_angle_y_mrad"),
            ),
        )
    )
    key_rows = _table_rows(
        (
            ("X minimum diameter (um)", key_x.get("minimum_diameter_um")),
            ("X minimum z (mm)", key_x.get("minimum_z_mm")),
            ("X minimum status", key_x.get("minimum_status")),
            ("X residual std (um)", key_x.get("residual_std_um")),
            ("X fit count", key_x.get("valid_points_count")),
            ("X measured full angle (mrad)", key_x.get("measured_full_angle_mrad")),
            (
                "X Fraunhofer full angle (mrad)",
                key_x.get("fraunhofer_ideal_full_angle_mrad"),
            ),
            ("Y minimum diameter (um)", key_y.get("minimum_diameter_um")),
            ("Y minimum z (mm)", key_y.get("minimum_z_mm")),
            ("Y minimum status", key_y.get("minimum_status")),
            ("Y residual std (um)", key_y.get("residual_std_um")),
            ("Y fit count", key_y.get("valid_points_count")),
            ("Y measured full angle (mrad)", key_y.get("measured_full_angle_mrad")),
            (
                "Y Fraunhofer full angle (mrad)",
                key_y.get("fraunhofer_ideal_full_angle_mrad"),
            ),
            ("Minimum-area z (mm)", key_area.get("z_mm")),
            ("Minimum-area X diameter (um)", key_area.get("diameter_x_um")),
            ("Minimum-area Y diameter (um)", key_area.get("diameter_y_um")),
            ("Minimum ellipse area (um2)", key_area.get("area_um2")),
        )
    )
    appendix_images = "\n".join(
        _path_image_figure(item)
        for item in beam_spot_images
        if isinstance(item, Mapping)
    )
    key_images = "\n".join(
        _path_image_figure(item) for item in context.get("key_spot_images", [])
        if isinstance(item, Mapping)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Laser Beam QA Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; }}
    h1, h2 {{ margin-bottom: 0.35rem; }}
    table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 720px; width: 100%; border: 1px solid #d1d5db; }}
    figure img {{ image-rendering: pixelated; }}
    .judgement {{ font-size: 1.2rem; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Laser Beam QA Report</h1>
  <p class="judgement">Final judgement: {html.escape(str(final_judgement))}</p>
  <p>Measurement: {html.escape(str(measurement))}</p>
  <p>{html.escape(model_note)}</p>
  <table>
    <tr><th>Sample ID</th><td>{html.escape(str(sample_info.get("sample_id", "")))}</td></tr>
    <tr><th>Recipe</th><td>{html.escape(str(recipe.get("recipe_id", "")))}</td></tr>
    <tr><th>Mode</th><td>{html.escape(str(system.get("mode", "")))}</td></tr>
  </table>
  <h2>Divergence</h2>
  <table>
    {divergence_rows}
  </table>
  <h2>Highlighted Results</h2>
  <table>
    {key_rows}
  </table>
  {key_images}
  <h2>D4sigma Morphology</h2>
  <table>
    {morphology_rows}
  </table>
  <h2>{'Observed-Minimum Prediction' if linear else 'Gaussian Waist Prediction'}</h2>
  <table>
    {prediction_rows}
  </table>
  {_image_section("Near-field image", near_field_image_data_uri)}
  {_image_section("Z-scan fit", zscan_plot_data_uri)}
  {_image_section("Beam centroid and X/Y propagation", beam_propagation_plot_data_uri)}
  <h2>Errors</h2>
  <pre>{html.escape(_json_like(errors))}</pre>
  <h2>Output Files</h2>
  <pre>{html.escape(_json_like(dict(output_files)))}</pre>
  <details class="point-appendix"><summary>Appendix: All Scan-Point Beam Images</summary>
  {appendix_images}
  </details>
</body>
</html>
"""


def _table_rows(rows: Sequence[tuple[str, Any]]) -> str:
    return "\n    ".join(
        f"<tr><th>{html.escape(label)}</th><td>{_format_html_value(value)}</td></tr>"
        for label, value in rows
    )


def _image_section(title: str, data_uri: Any) -> str:
    if not isinstance(data_uri, str) or not data_uri:
        return ""
    escaped_title = html.escape(title)
    escaped_data_uri = html.escape(data_uri)
    return (
        f'<h2>{escaped_title}</h2>'
        f'<img src="{escaped_data_uri}" alt="{escaped_title}">'
    )


def _path_image_figure(item: Mapping[str, Any]) -> str:
    source = html.escape(str(item.get("src", "")))
    caption = html.escape(str(item.get("caption", "")))
    if not source:
        return ""
    return (
        f'<figure><img src="{source}" alt="scan point">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _format_html_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return html.escape(f"{value:.6g}")
    return html.escape(str(value))


def _json_like(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


__all__ = ["ReportGenerator", "render_beam_propagation_image"]
