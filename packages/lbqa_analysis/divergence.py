"""D4sigma z-scan divergence fitting."""

from __future__ import annotations

import math
from collections.abc import Sequence

from lbqa_contracts.schemas import DivergenceFitResult, ZScanPlaneRecord


def fit_axis_diameter_squared(
    z_actual_mm: Sequence[float], diameter_um: Sequence[float]
) -> tuple[float, float, float]:
    """Fit D2(z) = a*z2 + b*z + c for one axis.

    The returned coefficients use:
    - a: um2/mm2
    - b: um2/mm
    - c: um2
    """

    if len(z_actual_mm) != len(diameter_um):
        raise ValueError("z_actual_mm and diameter_um must have the same length.")
    if len(z_actual_mm) < 3:
        raise ValueError("At least 3 z planes are required for a quadratic fit.")

    rows = [(z * z, z, 1.0) for z in z_actual_mm]
    y_values = [diameter * diameter for diameter in diameter_um]

    normal_matrix = [[0.0 for _ in range(3)] for _ in range(3)]
    normal_vector = [0.0 for _ in range(3)]
    for row, y_value in zip(rows, y_values, strict=True):
        for i in range(3):
            normal_vector[i] += row[i] * y_value
            for j in range(3):
                normal_matrix[i][j] += row[i] * row[j]

    a_um2_per_mm2, b_um2_per_mm, c_um2 = _solve_3x3(normal_matrix, normal_vector)
    return a_um2_per_mm2, b_um2_per_mm, c_um2


def fit_divergence_from_planes(planes: Sequence[ZScanPlaneRecord]) -> DivergenceFitResult:
    """Fit x/y divergence from z-scan plane records."""

    z_actual_mm = [plane.z_actual_mm for plane in planes]
    d4sigma_x_um = [plane.d4sigma_x_um for plane in planes]
    d4sigma_y_um = [plane.d4sigma_y_um for plane in planes]

    fit_a_x_um2_per_mm2, fit_b_x_um2_per_mm, fit_c_x_um2 = fit_axis_diameter_squared(
        z_actual_mm, d4sigma_x_um
    )
    fit_a_y_um2_per_mm2, fit_b_y_um2_per_mm, fit_c_y_um2 = fit_axis_diameter_squared(
        z_actual_mm, d4sigma_y_um
    )

    waist_z_x_mm, waist_d4sigma_x_um = _waist_from_quadratic(
        fit_a_x_um2_per_mm2, fit_b_x_um2_per_mm, fit_c_x_um2
    )
    waist_z_y_mm, waist_d4sigma_y_um = _waist_from_quadratic(
        fit_a_y_um2_per_mm2, fit_b_y_um2_per_mm, fit_c_y_um2
    )

    full_angle_x_mrad = math.sqrt(max(fit_a_x_um2_per_mm2, 0.0))
    full_angle_y_mrad = math.sqrt(max(fit_a_y_um2_per_mm2, 0.0))

    return DivergenceFitResult(
        sample_count=len(planes),
        fit_a_x_um2_per_mm2=fit_a_x_um2_per_mm2,
        fit_b_x_um2_per_mm=fit_b_x_um2_per_mm,
        fit_c_x_um2=fit_c_x_um2,
        fit_a_y_um2_per_mm2=fit_a_y_um2_per_mm2,
        fit_b_y_um2_per_mm=fit_b_y_um2_per_mm,
        fit_c_y_um2=fit_c_y_um2,
        waist_z_x_mm=waist_z_x_mm,
        waist_z_y_mm=waist_z_y_mm,
        waist_d4sigma_x_um=waist_d4sigma_x_um,
        waist_d4sigma_y_um=waist_d4sigma_y_um,
        full_angle_x_mrad=full_angle_x_mrad,
        half_angle_x_mrad=full_angle_x_mrad / 2.0,
        full_angle_y_mrad=full_angle_y_mrad,
        half_angle_y_mrad=full_angle_y_mrad / 2.0,
    )


def _waist_from_quadratic(
    a_um2_per_mm2: float, b_um2_per_mm: float, c_um2: float
) -> tuple[float, float]:
    if abs(a_um2_per_mm2) < 1e-12:
        return 0.0, math.sqrt(max(c_um2, 0.0))
    waist_z_mm = -b_um2_per_mm / (2.0 * a_um2_per_mm2)
    waist_d2_um2 = c_um2 - (b_um2_per_mm**2 / (4.0 * a_um2_per_mm2))
    return waist_z_mm, math.sqrt(max(waist_d2_um2, 0.0))


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]

    for pivot_index in range(3):
        pivot_row = max(
            range(pivot_index, 3),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("Quadratic fit is singular; choose more varied z planes.")
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, 4):
            augmented[pivot_index][column_index] /= pivot_value

        for row_index in range(3):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            for column_index in range(pivot_index, 4):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]

    return augmented[0][3], augmented[1][3], augmented[2][3]
