"""Pre-scan interpretation and z-scan point planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import BeamPlaneResult

from lbqa_sequencer.commands import ZScanRecipe

ScanDirection = Literal["ascending", "descending"]


@dataclass(frozen=True, slots=True)
class ZScanPreScanSample:
    z_actual_mm: float
    d4sigma_x_um: float
    d4sigma_y_um: float
    valid: bool
    invalid_reason: str | None = None

    @classmethod
    def from_plane_result(cls, plane_result: BeamPlaneResult) -> ZScanPreScanSample:
        return cls(
            z_actual_mm=plane_result.z_actual_mm,
            d4sigma_x_um=plane_result.d4sigma_x_um,
            d4sigma_y_um=plane_result.d4sigma_y_um,
            valid=plane_result.valid,
            invalid_reason=plane_result.invalid_reason,
        )

    @property
    def max_diameter_um(self) -> float:
        return max(self.d4sigma_x_um, self.d4sigma_y_um)


@dataclass(frozen=True, slots=True)
class ZScanPreScanResult:
    samples: tuple[ZScanPreScanSample, ...]
    max_diameter_um: float
    diameter_slope_um_per_mm: float
    valid_samples_count: int


@dataclass(frozen=True, slots=True)
class ZScanPlan:
    z_positions_mm: tuple[float, ...]
    scan_direction: ScanDirection
    risk_flags: tuple[str, ...]
    scan_half_range_um: float
    step_um: float
    estimated_max_diameter_um: float
    field_of_view_limit_um: float
    prescan_result: ZScanPreScanResult


def estimate_prescan_result(samples: list[ZScanPreScanSample]) -> ZScanPreScanResult:
    valid_samples = [
        sample
        for sample in samples
        if sample.valid
        and _positive_finite(sample.d4sigma_x_um)
        and _positive_finite(sample.d4sigma_y_um)
    ]
    if len(valid_samples) < 2:
        raise LBQAError(
            ErrorCode.E_ANALYSIS_ZSCAN_TOO_FEW_POINTS,
            "pre-scan needs at least two valid beam plane results.",
        )

    sorted_samples = sorted(valid_samples, key=lambda sample: sample.z_actual_mm)
    max_diameter_um = max(sample.max_diameter_um for sample in sorted_samples)
    slope_um_per_mm = 0.0
    for left, right in zip(sorted_samples, sorted_samples[1:], strict=False):
        dz_mm = abs(right.z_actual_mm - left.z_actual_mm)
        if dz_mm <= 0.0:
            continue
        diameter_change_um = abs(right.max_diameter_um - left.max_diameter_um)
        slope_um_per_mm = max(slope_um_per_mm, diameter_change_um / dz_mm)

    return ZScanPreScanResult(
        samples=tuple(samples),
        max_diameter_um=max_diameter_um,
        diameter_slope_um_per_mm=slope_um_per_mm,
        valid_samples_count=len(valid_samples),
    )


def build_zscan_plan(
    recipe: ZScanRecipe,
    prescan_result: ZScanPreScanResult,
    *,
    field_of_view_x_um: float,
    field_of_view_y_um: float,
) -> ZScanPlan:
    """Build centered z positions from recipe constraints and pre-scan risk."""

    _validate_field_of_view(field_of_view_x_um, field_of_view_y_um)
    risk_flags: list[str] = []
    field_of_view_limit_um = (
        min(field_of_view_x_um, field_of_view_y_um) * recipe.field_of_view_margin_percent / 100.0
    )
    max_half_range_um = recipe.max_scan_half_range_um
    slope_um_per_mm = max(0.0, prescan_result.diameter_slope_um_per_mm)
    estimated_max_diameter_um = _estimated_diameter_at_range(
        prescan_result.max_diameter_um,
        slope_um_per_mm,
        max_half_range_um,
    )

    if prescan_result.max_diameter_um >= field_of_view_limit_um:
        raise LBQAError(
            ErrorCode.E_SAFETY_COLLISION_RISK,
            "pre-scan beam diameter already exceeds the configured field-of-view margin.",
        )

    scan_half_range_um = max_half_range_um
    if estimated_max_diameter_um > field_of_view_limit_um:
        if not recipe.allow_range_shrink or slope_um_per_mm <= 0.0:
            raise LBQAError(
                ErrorCode.E_SAFETY_COLLISION_RISK,
                "requested z-scan range risks clipping the beam outside the field of view.",
            )
        safe_half_range_um = (
            (field_of_view_limit_um - prescan_result.max_diameter_um)
            / slope_um_per_mm
            * 1000.0
        )
        scan_half_range_um = max(0.0, min(max_half_range_um, safe_half_range_um))
        estimated_max_diameter_um = _estimated_diameter_at_range(
            prescan_result.max_diameter_um,
            slope_um_per_mm,
            scan_half_range_um,
        )
        risk_flags.append("range_shrunk_for_field_of_view")

    if scan_half_range_um <= 0.0:
        raise LBQAError(
            ErrorCode.E_SAFETY_COLLISION_RISK,
            "safe z-scan half-range collapsed to zero during planning.",
        )

    if estimated_max_diameter_um > field_of_view_limit_um * 0.9:
        risk_flags.append("near_field_of_view_margin")

    step_um = _target_step_um(recipe, slope_um_per_mm, scan_half_range_um)
    points_count = _points_count(
        scan_half_range_um=scan_half_range_um,
        step_um=step_um,
        min_points=recipe.min_points,
        max_points=recipe.max_points,
    )
    z_positions_mm = _centered_positions_mm(
        z_ref_mm=recipe.z_ref_mm,
        scan_half_range_um=scan_half_range_um,
        points_count=points_count,
    )
    if len(set(z_positions_mm)) < recipe.min_points:
        raise LBQAError(
            ErrorCode.E_ANALYSIS_ZSCAN_TOO_FEW_POINTS,
            "z-scan plan could not produce enough unique z positions.",
        )

    return ZScanPlan(
        z_positions_mm=z_positions_mm,
        scan_direction="ascending",
        risk_flags=tuple(risk_flags),
        scan_half_range_um=scan_half_range_um,
        step_um=step_um,
        estimated_max_diameter_um=estimated_max_diameter_um,
        field_of_view_limit_um=field_of_view_limit_um,
        prescan_result=prescan_result,
    )


def _target_step_um(
    recipe: ZScanRecipe,
    slope_um_per_mm: float,
    scan_half_range_um: float,
) -> float:
    if slope_um_per_mm <= 1e-12:
        return (2.0 * scan_half_range_um) / float(recipe.min_points - 1)
    target_step_um = recipe.target_diameter_change_per_step_um / slope_um_per_mm * 1000.0
    maximum_step_um = (2.0 * scan_half_range_um) / float(recipe.min_points - 1)
    return min(max(target_step_um, 1e-6), maximum_step_um)


def _points_count(
    *,
    scan_half_range_um: float,
    step_um: float,
    min_points: int,
    max_points: int,
) -> int:
    estimated_count = int(math.floor((2.0 * scan_half_range_um) / step_um)) + 1
    points_count = max(min_points, min(max_points, estimated_count))
    if points_count % 2 == 0:
        if points_count < max_points:
            points_count += 1
        else:
            points_count -= 1
    return max(min_points, points_count)


def _centered_positions_mm(
    *,
    z_ref_mm: float,
    scan_half_range_um: float,
    points_count: int,
) -> tuple[float, ...]:
    start_mm = z_ref_mm - scan_half_range_um / 1000.0
    stop_mm = z_ref_mm + scan_half_range_um / 1000.0
    if points_count == 1:
        return (z_ref_mm,)
    step_mm = (stop_mm - start_mm) / float(points_count - 1)
    return tuple(start_mm + index * step_mm for index in range(points_count))


def _estimated_diameter_at_range(
    max_prescan_diameter_um: float,
    slope_um_per_mm: float,
    scan_half_range_um: float,
) -> float:
    return max_prescan_diameter_um + slope_um_per_mm * scan_half_range_um / 1000.0


def _validate_field_of_view(field_of_view_x_um: float, field_of_view_y_um: float) -> None:
    if field_of_view_x_um <= 0.0 or field_of_view_y_um <= 0.0:
        raise ValueError("field_of_view_x_um and field_of_view_y_um must be positive.")


def _positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


__all__ = [
    "ScanDirection",
    "ZScanPlan",
    "ZScanPreScanResult",
    "ZScanPreScanSample",
    "build_zscan_plan",
    "estimate_prescan_result",
]
