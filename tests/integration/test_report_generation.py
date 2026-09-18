from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from lbqa_analysis.far_field_fit import fit_far_field
from lbqa_contracts.models import BeamPlaneResult, ZScanResult
from lbqa_data import RunStore
from lbqa_data import report_generator as report
from PIL import Image


def fixed_timestamp() -> datetime:
    return datetime(2026, 6, 19, 10, 11, 12, tzinfo=UTC)


def plane(z_actual_mm: float) -> BeamPlaneResult:
    return BeamPlaneResult(
        z_actual_mm=z_actual_mm,
        centroid_x_um=12.0,
        centroid_y_um=13.0,
        peak_x_um=12.0,
        peak_y_um=13.0,
        d4sigma_x_um=18.0 + abs(z_actual_mm) * 2.0,
        d4sigma_y_um=22.0 + abs(z_actual_mm) * 2.5,
        d4sigma_major_um=22.0 + abs(z_actual_mm) * 2.5,
        d4sigma_minor_um=18.0 + abs(z_actual_mm) * 2.0,
        fwhm_x_um=9.0,
        fwhm_y_um=11.0,
        ellipticity=1.2,
        azimuth_deg=0.0,
        peak_value=3000.0,
        saturation_pixels=0,
        edge_energy_percent=0.2,
        valid=True,
        invalid_reason=None,
    )


def zscan(points: list[BeamPlaneResult]) -> ZScanResult:
    return ZScanResult(
        points=points,
        full_angle_x_mrad=8.0,
        full_angle_y_mrad=10.0,
        half_angle_x_mrad=4.0,
        half_angle_y_mrad=5.0,
        waist_z_x_mm=0.0,
        waist_z_y_mm=0.0,
        waist_diameter_x_um=18.0,
        waist_diameter_y_um=22.0,
        fit_r2_x=0.995,
        fit_r2_y=0.994,
        valid_points_count=len(points),
        judgement="pass",
        invalid_reason=None,
    )


def test_run_store_generates_traceable_report_artifacts(tmp_path) -> None:
    store = RunStore(tmp_path, timestamp_factory=fixed_timestamp)
    record = store.create_run(
        "sample-integration",
        {"recipe_id": "recipe-40x-zscan", "operator_mode": "simulated"},
    )
    store.save_calibration_snapshot({"calibration_id": "cal-40x", "magnification": 40.0})
    store.save_system_snapshot({"system_name": "laser-beam-qa", "default_mode": "simulated"})
    assert store.image_store is not None

    points = [plane(-0.2), plane(0.0), plane(0.2)]
    for index, point in enumerate(points):
        store.image_store.save_report_source_npy(
            np.full((16, 16), float(index), dtype=np.float32),
            frame_index_count=index,
        )
        store.save_plane_result(
            plane_result=point,
            z_cmd_mm=point.z_actual_mm,
            exposure_us=2000.0,
            frame_count=3,
        )

    summary = store.save_zscan_result(zscan_result=zscan(points))
    assert summary.exists()
    final_summary = store.finalize_run(render_html=True)

    rows = list(csv.DictReader((record.run_dir / "z_scan_table.csv").open(encoding="utf-8")))
    loaded_summary = json.loads((record.run_dir / "result_summary.json").read_text("utf-8"))
    report_html = (record.run_dir / "report.html").read_text("utf-8")

    assert len(rows) == 3
    assert loaded_summary == final_summary
    assert loaded_summary["output_files"]["z_scan_table"] == "z_scan_table.csv"
    assert loaded_summary["output_files"]["result_summary_csv"] == "result_summary.csv"
    assert "images" not in loaded_summary["output_files"]
    assert "processed" not in loaded_summary["output_files"]
    assert "sample-integration" in report_html
    assert "D(z)=sqrt(D0^2+Theta^2(z-z0)^2)" in report_html
    assert "双轴发散拟合" in report_html
    assert "附录：全部扫描点模斑图" in report_html
    assert (record.run_dir / "report_assets" / "beam_propagation.png").stat().st_size > 0
    assert not list(record.run_dir.rglob("*.npy"))
    assert not list(record.run_dir.rglob("*.tif"))
    assert not list(record.run_dir.rglob("*.tiff"))


class ReportMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.details = []
        self.appendix_images = []
        self.collapsed_appendix = False
        self.captions = []
        self.in_caption = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "details":
            appendix = attributes.get("class") == "point-appendix"
            self.details.append(appendix)
            if appendix:
                self.collapsed_appendix = "open" not in attributes
        if tag == "img" and any(self.details):
            self.appendix_images.append(attributes["src"])
        if tag == "figcaption":
            self.in_caption = True

    def handle_endtag(self, tag):
        if tag == "details":
            self.details.pop()
        if tag == "figcaption":
            self.in_caption = False

    def handle_data(self, data):
        if self.in_caption:
            self.captions.append(data)


@pytest.mark.parametrize("template_mode", ["source", "bundled", "fallback"])
def test_linear_report_and_assets_are_consistent_with_review(
    tmp_path, monkeypatch, template_mode,
):
    generator = report.ReportGenerator()
    if template_mode == "bundled":
        template_dir = tmp_path / "_internal" / "lbqa_data" / "report_templates"
        shutil.copytree(Path(report.__file__).with_name("report_templates"), template_dir)
        generator = report.ReportGenerator(template_dir=template_dir)
    elif template_mode == "fallback":
        generator._environment = None
    store = RunStore(tmp_path / "runs", report_generator=generator)
    record = store.create_run("linear-operator", {
        "recipe_id": "operator", "gaussian_prediction": {"wavelength_nm": 1064.0},
    })
    store.save_system_snapshot({
        "effective_pixel_x_um": 0.5, "effective_pixel_y_um": 0.5,
        "visualization": {"colormap": "gray", "overlay_mode": "none"},
    })
    points = [replace(
        plane(float(z)), d4sigma_x_um=10.0 + z, d4sigma_y_um=18.0 + 2 * z,
        azimuth_deg=15.0, width_source="gaussian_fit_continuous_axes",
    ) for z in range(5)]
    fine = replace(points[0], z_actual_mm=0.5, d4sigma_x_um=7.0, d4sigma_y_um=8.0)
    image_y, image_x = np.indices((256, 256))
    spot = np.exp(-0.5 * (((image_x - 24) / 4)**2 + ((image_y - 26) / 6)**2))
    for index, point in enumerate([*points, fine]):
        store.save_plane_result(
            plane_result=point, phase="fine" if index == 5 else "custom",
            axis_x_azimuth_deg=15.0, native_major_azimuth_deg=105.0,
            axis_orientation_uncertain=False,
        )
        store.image_store.save_report_source_npy(spot * (index + 1), frame_index_count=index)
    review_dir = record.run_dir / ".working" / "review_points"
    review_dir.mkdir()
    Image.new("RGB", (4, 4), "red").save(review_dir / "point_000000.png")
    options_seen = []
    axes_seen = []
    titles_seen = []
    original = report.save_beam_spot_png

    def save_spot(image, path, **kwargs):
        options_seen.append(kwargs["options"])
        axes_seen.append(kwargs["axes"])
        titles_seen.append(kwargs["title"])
        original(image, path, **kwargs)

    monkeypatch.setattr(report, "save_beam_spot_png", save_spot)
    result = fit_far_field(points, selected_indices_x=[1, 2], selected_indices_y=[2, 3, 4])
    preview = store.preview_summary(result)
    store.save_zscan_result(zscan_result=result)
    final = store.finalize_run()
    assert final["key_results"] == preview["key_results"]
    markup = (record.run_dir / "report.html").read_text(encoding="utf-8")
    parser = ReportMarkup()
    parser.feed(markup)
    assert parser.collapsed_appendix
    assert len(parser.appendix_images) == 6
    assert len(set(parser.appendix_images)) == 6
    assert "D(z)=sqrt" not in markup
    assert "D(z)=intercept_at_reference+slope*(z-z_reference)" in markup
    assert "image-rendering: pixelated" in markup
    assert "linear-operator" in markup
    assert "PASS" not in markup.split("</style>")[-1]
    assert "width X/Y=7/8 um" in markup
    assert "X-axis angle=15.00 deg" in markup
    assert len(options_seen) == 6
    assert len({(value.intensity_min, value.intensity_max) for value in options_seen}) == 1
    assert all(value.zoom_factor > 1 and value.colormap == "thermal" for value in options_seen)
    assert [value.x_axis_azimuth_deg for value in axes_seen] == pytest.approx([15.0] * 6)
    assert [value.azimuth_deg for value in axes_seen] == pytest.approx([105.0] * 6)
    assert [value.major_sigma_px for value in axes_seen] == pytest.approx([
        value.d4sigma_major_um / 2.0 for value in [*points, fine]
    ])
    assert [value.minor_sigma_px for value in axes_seen] == pytest.approx([
        value.d4sigma_minor_um / 2.0 for value in [*points, fine]
    ])
    assert all(value.x_axis_sigma_px < value.y_axis_sigma_px for value in axes_seen)
    assert all("X=" in value and "Y=" in value for value in titles_seen)
    for src in parser.appendix_images:
        with Image.open(record.run_dir / src) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert np.ptp(pixels) > 0
        assert np.any(pixels[:, :, 0] != pixels[:, :, 1])
    assert not (record.run_dir / ".working").exists()
    assert not list(record.run_dir.rglob("*.npy"))
    assert not list(record.run_dir.rglob("*.tif*"))
    assert not list(record.run_dir.rglob("point_*.png"))
    assert final["key_results"]["x"]["residual_std_um"] is None
    assert final["key_results"]["x"]["valid_points_count"] == 2
    assert final["key_results"]["y"]["valid_points_count"] == 3
