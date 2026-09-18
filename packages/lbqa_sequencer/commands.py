"""Command objects consumed by the sequencer service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

InvalidPointPolicy = Literal["skip", "retry", "stop"]
OperatorMode = Literal["automatic", "semi_auto", "simulated"]


@dataclass(frozen=True, slots=True)
class ZScanRecipe:
    recipe_id: str
    z_ref_mm: float
    prescan_delta_um: float
    min_points: int
    max_points: int
    max_scan_half_range_um: float
    target_diameter_change_per_step_um: float
    operator_mode: OperatorMode = "automatic"
    dark_frame_enabled: bool = False
    dark_frame_count: int | None = None
    auto_exposure_enabled: bool = True
    auto_align_enabled: bool = True
    manual_align_required: bool = False
    require_home_before_scan: bool = True
    average_frame_count: int | None = None
    stage_timeout_s: float = 5.0
    stage_tolerance_um: float = 1.0
    field_of_view_margin_percent: float = 85.0
    allow_range_shrink: bool = True
    invalid_point_policy: InvalidPointPolicy = "skip"
    retry_limit_count: int = 0
    max_invalid_points: int | None = None
    soft_limit_min_mm: float | None = None
    soft_limit_max_mm: float | None = None
    quality_limits: Mapping[str, Any] = field(default_factory=dict)
    judgement_limits: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.recipe_id:
            raise ValueError("recipe_id is required.")
        if self.operator_mode not in {"automatic", "semi_auto", "simulated"}:
            raise ValueError("operator_mode must be automatic, semi_auto, or simulated.")
        if self.prescan_delta_um <= 0.0:
            raise ValueError("prescan_delta_um must be positive.")
        minimum = 2 if self.quality_limits.get("fit_model") == "far_field_linear" else 3
        if self.min_points < minimum:
            raise ValueError(f"min_points must be at least {minimum} for this fit model.")
        if self.max_points < self.min_points:
            raise ValueError("max_points must be greater than or equal to min_points.")
        if self.max_scan_half_range_um <= 0.0:
            raise ValueError("max_scan_half_range_um must be positive.")
        if self.target_diameter_change_per_step_um <= 0.0:
            raise ValueError("target_diameter_change_per_step_um must be positive.")
        if self.stage_timeout_s <= 0.0:
            raise ValueError("stage_timeout_s must be positive.")
        if self.stage_tolerance_um < 0.0:
            raise ValueError("stage_tolerance_um must be non-negative.")
        if not 0.0 < self.field_of_view_margin_percent <= 100.0:
            raise ValueError("field_of_view_margin_percent must be in (0, 100].")
        if self.retry_limit_count < 0:
            raise ValueError("retry_limit_count must be non-negative.")
        if self.max_invalid_points is not None and self.max_invalid_points < 0:
            raise ValueError("max_invalid_points must be non-negative when provided.")
        if self.invalid_point_policy not in {"skip", "retry", "stop"}:
            raise ValueError("invalid_point_policy must be skip, retry, or stop.")
        if (
            self.soft_limit_min_mm is not None
            and self.soft_limit_max_mm is not None
            and self.soft_limit_min_mm >= self.soft_limit_max_mm
        ):
            raise ValueError("soft_limit_min_mm must be less than soft_limit_max_mm.")

    @classmethod
    def from_mapping(cls, recipe: Mapping[str, Any]) -> ZScanRecipe:
        """Build a z-scan recipe from a flat or nested mapping."""

        zscan = _mapping(recipe.get("z_scan"))
        scan = _mapping(recipe.get("scan"))
        safety = _mapping(recipe.get("safety"))
        camera = _mapping(recipe.get("camera"))
        analysis = _mapping(recipe.get("analysis"))
        limits = _mapping(recipe.get("limits"))

        z_ref_mm = _number(recipe, zscan, scan, keys=("z_ref_mm",), default=0.0)
        max_scan_half_range_um = _number(
            recipe,
            zscan,
            keys=("max_scan_half_range_um",),
            default=_half_range_from_scan_um(scan),
        )
        return cls(
            recipe_id=str(recipe.get("recipe_id", "zscan_recipe")),
            operator_mode=str(recipe.get("operator_mode", recipe.get("mode", "automatic"))),
            z_ref_mm=z_ref_mm,
            prescan_delta_um=_number(
                recipe,
                zscan,
                keys=("prescan_delta_um",),
                default=50.0,
            ),
            min_points=int(_number(recipe, zscan, keys=("min_points",), default=5)),
            max_points=int(_number(recipe, zscan, keys=("max_points",), default=11)),
            max_scan_half_range_um=max_scan_half_range_um,
            target_diameter_change_per_step_um=_number(
                recipe,
                zscan,
                keys=("target_diameter_change_per_step_um",),
                default=1.0,
            ),
            dark_frame_enabled=bool(zscan.get("dark_frame_enabled", False)),
            dark_frame_count=_optional_int(zscan.get("dark_frame_count")),
            auto_exposure_enabled=bool(zscan.get("auto_exposure_enabled", True)),
            auto_align_enabled=bool(zscan.get("auto_align_enabled", True)),
            manual_align_required=bool(zscan.get("manual_align_required", False)),
            require_home_before_scan=bool(safety.get("require_home_before_scan", True)),
            average_frame_count=_optional_int(
                zscan.get("average_frame_count", camera.get("frame_average_count"))
            ),
            stage_timeout_s=_number(recipe, zscan, keys=("stage_timeout_s",), default=5.0),
            stage_tolerance_um=_number(
                recipe,
                zscan,
                keys=("stage_tolerance_um",),
                default=1.0,
            ),
            field_of_view_margin_percent=_number(
                recipe,
                zscan,
                keys=("field_of_view_margin_percent",),
                default=85.0,
            ),
            allow_range_shrink=bool(zscan.get("allow_range_shrink", True)),
            invalid_point_policy=str(zscan.get("invalid_point_policy", "skip")),
            retry_limit_count=int(zscan.get("retry_limit_count", 0)),
            max_invalid_points=_optional_int(zscan.get("max_invalid_points")),
            soft_limit_min_mm=_optional_float(safety.get("z_soft_min_mm")),
            soft_limit_max_mm=_optional_float(safety.get("z_soft_max_mm")),
            quality_limits=dict(analysis),
            judgement_limits=dict(limits),
        )


@dataclass(frozen=True, slots=True)
class ZScanRunCommand:
    run_id: str
    sample_id: str
    recipe: ZScanRecipe
    calibration: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required.")
        if not self.sample_id:
            raise ValueError("sample_id is required.")


class SequencerOperatorCommandType(StrEnum):
    ABORT = "abort"
    RETRY_POINT = "retry_point"
    SKIP_POINT = "skip_point"


@dataclass(frozen=True, slots=True)
class SequencerOperatorCommand:
    command_type: SequencerOperatorCommandType
    reason: str | None = None


class CommandQueue:
    """In-memory command queue used by UI or tests to request abort/retry/skip."""

    def __init__(self) -> None:
        self._commands: list[SequencerOperatorCommand] = []

    def push(self, command: SequencerOperatorCommand) -> None:
        self._commands.append(command)

    def request_abort(self, reason: str | None = None) -> None:
        self.push(SequencerOperatorCommand(SequencerOperatorCommandType.ABORT, reason=reason))

    def request_retry_point(self, reason: str | None = None) -> None:
        self.push(SequencerOperatorCommand(SequencerOperatorCommandType.RETRY_POINT, reason=reason))

    def request_skip_point(self, reason: str | None = None) -> None:
        self.push(SequencerOperatorCommand(SequencerOperatorCommandType.SKIP_POINT, reason=reason))

    def abort_requested(self) -> bool:
        return any(
            command.command_type == SequencerOperatorCommandType.ABORT for command in self._commands
        )

    def consume_point_command(self) -> SequencerOperatorCommand | None:
        for index, command in enumerate(self._commands):
            if command.command_type in {
                SequencerOperatorCommandType.RETRY_POINT,
                SequencerOperatorCommandType.SKIP_POINT,
            }:
                return self._commands.pop(index)
        return None

    def clear(self) -> None:
        self._commands.clear()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(
    *sources: Mapping[str, Any],
    keys: tuple[str, ...],
    default: float,
) -> float:
    for source in sources:
        for key in keys:
            if key in source:
                return float(source[key])
    return float(default)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _half_range_from_scan_um(scan: Mapping[str, Any]) -> float:
    if "z_targets_mm" in scan:
        targets = [float(value) for value in scan["z_targets_mm"]]
        if targets:
            return (max(targets) - min(targets)) * 500.0
    if "z_start_mm" in scan and "z_stop_mm" in scan:
        return abs(float(scan["z_stop_mm"]) - float(scan["z_start_mm"])) * 500.0
    return 500.0


__all__ = [
    "CommandQueue",
    "InvalidPointPolicy",
    "OperatorMode",
    "SequencerOperatorCommand",
    "SequencerOperatorCommandType",
    "ZScanRecipe",
    "ZScanRunCommand",
]
