"""Executable z-scan workflow coordinated by the sequencer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from lbqa_contracts.errors import ErrorCode, LBQAError
from lbqa_contracts.models import BeamPlaneResult, ZScanResult
from lbqa_contracts.ports import BeamProfilerPort, InterlockPort, StagePort

from lbqa_sequencer.commands import (
    CommandQueue,
    SequencerOperatorCommandType,
    ZScanRecipe,
    ZScanRunCommand,
)
from lbqa_sequencer.error_recovery import ErrorRecoveryService
from lbqa_sequencer.events import SequencerEvent, SequencerEventType
from lbqa_sequencer.state_machine import SequencerState, SequenceStateMachine
from lbqa_sequencer.zscan_plan import (
    ZScanPlan,
    ZScanPreScanResult,
    ZScanPreScanSample,
    build_zscan_plan,
    estimate_prescan_result,
)


class AcquisitionServicePort(Protocol):
    def prepare_dark_frame(self, frame_count: int | None = None) -> Any:
        """Capture or load a dark frame."""

    def prepare_exposure(self) -> Any:
        """Run auto exposure."""

    def capture_plane_average(self, z_actual_mm: float, frame_count: int | None = None) -> Any:
        """Capture an averaged plane at z_actual_mm."""


class AnalysisServicePort(Protocol):
    def analyze_beam_plane(
        self,
        image: Any,
        calibration: Any,
        z_actual_mm: float,
        quality_limits: dict[str, Any],
    ) -> BeamPlaneResult:
        """Analyze one acquired beam plane."""

    def fit_zscan(self, points: list[BeamPlaneResult]) -> ZScanResult:
        """Fit the z-scan divergence result."""

    def judge_result(self, zscan_result: ZScanResult, limits: dict[str, Any] | None = None) -> str:
        """Judge the fitted result."""


@dataclass(slots=True)
class SequencerDependencies:
    stage: StagePort
    profiler: BeamProfilerPort
    interlock: InterlockPort
    acquisition: AcquisitionServicePort
    analysis: AnalysisServicePort
    data_store: Any | None = None
    report_service: Any | None = None
    alignment_service: Any | None = None
    command_queue: CommandQueue = field(default_factory=CommandQueue)


@dataclass(slots=True)
class ZScanRunContext:
    command: ZScanRunCommand
    state_machine: SequenceStateMachine
    events: list[SequencerEvent]
    prescan_result: ZScanPreScanResult | None = None
    zscan_plan: ZScanPlan | None = None
    plane_results: list[BeamPlaneResult] = field(default_factory=list)
    zscan_result: ZScanResult | None = None
    errors: list[str] = field(default_factory=list)
    aborted: bool = False
    failure_exception: BaseException | None = None

    @property
    def recipe(self) -> ZScanRecipe:
        return self.command.recipe

    def failure_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.command.run_id,
            "sample_id": self.command.sample_id,
            "recipe_id": self.command.recipe.recipe_id,
            "state": self.state_machine.state.value,
            "errors": list(self.errors),
            "plane_results_count": len(self.plane_results),
            "z_positions_mm": (
                list(self.zscan_plan.z_positions_mm) if self.zscan_plan is not None else []
            ),
            "exception": str(self.failure_exception) if self.failure_exception else None,
        }


class ZScanSequence:
    """Stateful z-scan orchestration with recovery on any exception."""

    def __init__(self, dependencies: SequencerDependencies) -> None:
        self._deps = dependencies
        self.events: list[SequencerEvent] = []
        self.state_machine = SequenceStateMachine()

    def run(self, command: ZScanRunCommand) -> ZScanResult:
        context = ZScanRunContext(
            command=command,
            state_machine=self.state_machine,
            events=self.events,
        )
        try:
            self._run_normal_flow(context)
        except BaseException as exc:
            recovery = ErrorRecoveryService(
                state_machine=self.state_machine,
                stage=self._deps.stage,
                acquisition=self._deps.acquisition,
                data_store=self._deps.data_store,
                report_service=self._deps.report_service,
                events=self.events,
            )
            recovery.recover(context, exc)
            raise

        if context.zscan_result is None:
            raise RuntimeError("sequencer completed without a ZScanResult.")
        return context.zscan_result

    def _run_normal_flow(self, context: ZScanRunContext) -> None:
        self._enter(SequencerState.INITIALIZE, "initializing z-scan run")
        self._apply_recipe_soft_limits(context.recipe)
        self._abort_if_requested()

        self._enter(SequencerState.CONNECT_DEVICES, "connecting stage and profiler")
        self._deps.stage.connect()
        self._deps.stage.enable()
        self._deps.profiler.connect()
        self._abort_if_requested()

        self._enter(SequencerState.SAFETY_CHECK, "checking safety interlock")
        self._ensure_interlock_safe()

        self._enter(SequencerState.LOAD_RECIPE, "loading z-scan recipe")
        _ = context.recipe

        self._enter(SequencerState.HOME_STAGE, "homing z stage")
        if context.recipe.require_home_before_scan:
            self._deps.stage.home()
            if not self._deps.stage.is_homed():
                raise LBQAError(ErrorCode.E_DEVICE_STAGE_NOT_HOMED, "stage did not home.")

        self._enter(SequencerState.LOAD_CALIBRATION, "loading optical calibration")
        field_of_view_x_um, field_of_view_y_um = _field_of_view_um(context.command.calibration)

        self._enter(SequencerState.DARK_FRAME, "preparing dark frame")
        if context.recipe.dark_frame_enabled:
            self._deps.acquisition.prepare_dark_frame(context.recipe.dark_frame_count)

        align_state = (
            SequencerState.MANUAL_ALIGN
            if context.recipe.manual_align_required or context.recipe.operator_mode == "semi_auto"
            else SequencerState.AUTO_ALIGN
        )
        self._enter(align_state, f"{align_state.value} step")
        self._run_alignment(align_state)

        self._enter(SequencerState.AUTO_EXPOSURE, "preparing exposure")
        if context.recipe.auto_exposure_enabled:
            self._deps.acquisition.prepare_exposure()

        self._enter(SequencerState.PRE_SCAN_Z, "running z pre-scan")
        context.prescan_result = self._run_prescan(context)

        self._enter(SequencerState.BUILD_ZSCAN_PLAN, "building z-scan plan")
        context.zscan_plan = build_zscan_plan(
            context.recipe,
            context.prescan_result,
            field_of_view_x_um=field_of_view_x_um,
            field_of_view_y_um=field_of_view_y_um,
        )
        self._emit(
            SequencerEventType.PLAN_BUILT,
            "z-scan plan built",
            payload={
                "z_positions_mm": list(context.zscan_plan.z_positions_mm),
                "risk_flags": list(context.zscan_plan.risk_flags),
                "scan_half_range_um": context.zscan_plan.scan_half_range_um,
                "step_um": context.zscan_plan.step_um,
            },
        )

        self._enter(SequencerState.ZSCAN_LOOP, "capturing z-scan planes")
        context.plane_results = self._run_zscan_loop(context)

        self._enter(SequencerState.ANALYZE_ZSCAN, "fitting z-scan result")
        context.zscan_result = self._deps.analysis.fit_zscan(context.plane_results)

        self._enter(SequencerState.JUDGE_RESULT, "judging z-scan result")
        judgement = self._deps.analysis.judge_result(
            context.zscan_result,
            dict(context.recipe.judgement_limits),
        )
        context.zscan_result = _with_judgement(context.zscan_result, judgement)

        self._enter(SequencerState.SAVE_DATA, "saving z-scan result")
        _call_if_present(
            self._deps.data_store,
            "save_zscan_result",
            zscan_result=context.zscan_result,
            run_id=context.command.run_id,
            sample_id=context.command.sample_id,
        )

        self._enter(SequencerState.GENERATE_REPORT, "generating z-scan report")
        _call_if_present(
            self._deps.report_service,
            "generate_report",
            zscan_result=context.zscan_result,
            run_id=context.command.run_id,
            sample_id=context.command.sample_id,
        )

        self._enter(SequencerState.COMPLETE, "z-scan complete")
        self._emit(SequencerEventType.COMPLETE, "z-scan run completed")
        self.state_machine.transition_to(SequencerState.IDLE)

    def _run_prescan(self, context: ZScanRunContext) -> ZScanPreScanResult:
        delta_mm = context.recipe.prescan_delta_um / 1000.0
        z_positions_mm = (
            context.recipe.z_ref_mm - delta_mm,
            context.recipe.z_ref_mm,
            context.recipe.z_ref_mm + delta_mm,
        )
        samples: list[ZScanPreScanSample] = []
        for z_target_mm in z_positions_mm:
            plane_result = self._capture_and_analyze_plane(
                context,
                z_target_mm=z_target_mm,
                plane_index_count=len(samples),
                phase="prescan",
            )
            samples.append(ZScanPreScanSample.from_plane_result(plane_result))
        return estimate_prescan_result(samples)

    def _run_zscan_loop(self, context: ZScanRunContext) -> list[BeamPlaneResult]:
        if context.zscan_plan is None:
            raise RuntimeError("zscan_plan is required before ZScanLoop.")

        plane_results: list[BeamPlaneResult] = []
        invalid_points_count = 0
        for plane_index_count, z_target_mm in enumerate(context.zscan_plan.z_positions_mm):
            attempts_count = 0
            while True:
                operator_command = self._deps.command_queue.consume_point_command()
                if operator_command is not None:
                    self._emit(
                        SequencerEventType.COMMAND_ACCEPTED,
                        f"operator command: {operator_command.command_type.value}",
                        payload={"reason": operator_command.reason},
                    )
                    if operator_command.command_type == SequencerOperatorCommandType.SKIP_POINT:
                        break
                    if operator_command.command_type == SequencerOperatorCommandType.RETRY_POINT:
                        attempts_count = max(0, attempts_count - 1)

                plane_result = self._capture_and_analyze_plane(
                    context,
                    z_target_mm=z_target_mm,
                    plane_index_count=plane_index_count,
                    phase="zscan",
                )
                if plane_result.valid:
                    plane_results.append(plane_result)
                    break

                if (
                    context.recipe.invalid_point_policy == "retry"
                    and attempts_count < context.recipe.retry_limit_count
                ):
                    attempts_count += 1
                    self._emit(
                        SequencerEventType.POINT_INVALID,
                        "invalid z point will be retried",
                        error_code=plane_result.invalid_reason,
                        payload={
                            "plane_index_count": plane_index_count,
                            "z_target_mm": z_target_mm,
                            "attempts_count": attempts_count,
                        },
                    )
                    continue

                invalid_points_count += 1
                plane_results.append(plane_result)
                self._emit(
                    SequencerEventType.POINT_INVALID,
                    "invalid z point recorded",
                    error_code=plane_result.invalid_reason,
                    payload={
                        "plane_index_count": plane_index_count,
                        "z_target_mm": z_target_mm,
                    },
                )
                if context.recipe.invalid_point_policy == "stop":
                    raise LBQAError(
                        plane_result.invalid_reason or ErrorCode.E_ANALYSIS_D4SIGMA_FAILED,
                        "invalid z-scan point triggered stop policy.",
                    )
                if (
                    context.recipe.max_invalid_points is not None
                    and invalid_points_count > context.recipe.max_invalid_points
                ):
                    raise LBQAError(
                        ErrorCode.E_ANALYSIS_ZSCAN_TOO_FEW_POINTS,
                        "too many invalid z-scan points.",
                    )
                break
        return plane_results

    def _capture_and_analyze_plane(
        self,
        context: ZScanRunContext,
        *,
        z_target_mm: float,
        plane_index_count: int,
        phase: str,
    ) -> BeamPlaneResult:
        self._abort_if_requested()
        self._ensure_interlock_safe()
        self._deps.stage.move_abs_mm(z_target_mm)
        in_position = self._deps.stage.wait_in_position(
            context.recipe.stage_timeout_s,
            context.recipe.stage_tolerance_um,
        )
        if not in_position:
            raise LBQAError(
                ErrorCode.E_DEVICE_STAGE_MOVE_TIMEOUT,
                f"stage did not settle at z_target_mm={z_target_mm}.",
            )

        z_actual_mm = self._deps.stage.get_position_mm()
        frame = self._deps.acquisition.capture_plane_average(
            z_actual_mm=z_actual_mm,
            frame_count=context.recipe.average_frame_count,
        )
        plane_result = self._deps.analysis.analyze_beam_plane(
            image=frame.image,
            calibration=context.command.calibration,
            z_actual_mm=float(frame.z_actual_mm),
            quality_limits=dict(context.recipe.quality_limits),
        )
        _call_if_present(
            self._deps.data_store,
            "save_plane_result",
            plane_result=plane_result,
            plane_index_count=plane_index_count,
            z_target_mm=z_target_mm,
            phase=phase,
            run_id=context.command.run_id,
        )
        self._emit(
            SequencerEventType.PLANE_CAPTURED,
            f"{phase} plane captured",
            payload={
                "phase": phase,
                "plane_index_count": plane_index_count,
                "z_target_mm": z_target_mm,
                "z_actual_mm": plane_result.z_actual_mm,
                "valid": plane_result.valid,
                "invalid_reason": plane_result.invalid_reason,
            },
        )
        return plane_result

    def _run_alignment(self, state: SequencerState) -> None:
        if self._deps.alignment_service is None:
            return
        if state == SequencerState.MANUAL_ALIGN:
            _call_if_present(self._deps.alignment_service, "manual_align")
        else:
            _call_if_present(self._deps.alignment_service, "auto_align")

    def _apply_recipe_soft_limits(self, recipe: ZScanRecipe) -> None:
        if recipe.soft_limit_min_mm is not None and recipe.soft_limit_max_mm is not None:
            self._deps.stage.set_soft_limits(recipe.soft_limit_min_mm, recipe.soft_limit_max_mm)

    def _ensure_interlock_safe(self) -> None:
        if not self._deps.interlock.is_safe():
            status = self._deps.interlock.get_status()
            reason = status.get("reason", "safety interlock is unsafe")
            raise LBQAError(ErrorCode.E_SAFETY_INTERLOCK_OPEN, str(reason))

    def _abort_if_requested(self) -> None:
        if self._deps.command_queue.abort_requested():
            raise LBQAError(ErrorCode.E_USER_ABORT, "operator aborted z-scan sequence.")

    def _enter(self, state: SequencerState, message: str) -> None:
        self.state_machine.transition_to(state)
        self._emit(SequencerEventType.STATE_CHANGED, message)

    def _emit(
        self,
        event_type: SequencerEventType,
        message: str,
        *,
        error_code: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            SequencerEvent(
                event_type=event_type,
                state=self.state_machine.state,
                message=message,
                error_code=error_code,
                payload=payload or {},
            )
        )


def _field_of_view_um(calibration: Any) -> tuple[float, float]:
    field_of_view_x_um = _read_field(calibration, "field_of_view_x_um", None)
    field_of_view_y_um = _read_field(calibration, "field_of_view_y_um", None)
    if field_of_view_x_um is not None and field_of_view_y_um is not None:
        return float(field_of_view_x_um), float(field_of_view_y_um)

    effective_pixel_x_um = _read_field(calibration, "effective_pixel_x_um", None)
    effective_pixel_y_um = _read_field(calibration, "effective_pixel_y_um", None)
    width_px = _read_field(calibration, "width_px", None)
    height_px = _read_field(calibration, "height_px", None)
    if None not in {effective_pixel_x_um, effective_pixel_y_um, width_px, height_px}:
        return (
            float(effective_pixel_x_um) * float(width_px),
            float(effective_pixel_y_um) * float(height_px),
        )
    raise ValueError(
        "calibration must provide field_of_view_x_um/field_of_view_y_um "
        "or effective pixel sizes plus width_px/height_px."
    )


def _read_field(source: Any, field_name: str, default: Any) -> Any:
    if isinstance(source, dict):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


def _with_judgement(zscan_result: ZScanResult, judgement: str) -> ZScanResult:
    normalized = judgement.lower()
    if normalized == "pass":
        return replace(zscan_result, judgement="pass", invalid_reason=None)
    if normalized == "fail":
        return replace(
            zscan_result,
            judgement="fail",
            invalid_reason=zscan_result.invalid_reason or "quality_limit_failed",
        )
    if normalized == "invalid":
        return replace(
            zscan_result,
            judgement="fail",
            invalid_reason=zscan_result.invalid_reason
            or ErrorCode.E_ANALYSIS_ZSCAN_FIT_FAILED.value,
        )
    return replace(zscan_result, judgement="unknown")


def _call_if_present(target: Any | None, method_name: str, **kwargs: Any) -> Any:
    if target is None:
        return None
    method = getattr(target, method_name, None)
    if method is None:
        return None
    return method(**kwargs)


__all__ = [
    "AcquisitionServicePort",
    "AnalysisServicePort",
    "SequencerDependencies",
    "ZScanRunContext",
    "ZScanSequence",
]
