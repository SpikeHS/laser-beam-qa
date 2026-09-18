"""Helpers for unit-bearing field-name validation."""

UNIT_SUFFIXES = (
    "_mm",
    "_um",
    "_mrad",
    "_us",
    "_deg",
    "_px",
    "_pixels",
    "_ms",
    "_s",
    "_count",
    "_percent",
    "_ratio",
    "_db",
    "_iso",
    "_date",
    "_id",
    "_dir",
    "_um_per_px",
    "_mm_per_s",
    "_um2",
    "_um2_per_mm",
    "_um2_per_mm2",
)

DIMENSIONLESS_FIELD_NAMES = {
    "analysis",
    "bit_depth",
    "camera",
    "camera_model",
    "connected",
    "enabled",
    "error_code",
    "errors",
    "gain",
    "homed",
    "image",
    "invalid_reason",
    "judgement",
    "ellipticity",
    "fit_r2_x",
    "fit_r2_y",
    "fit_diagnostics",
    "limits",
    "low_signal",
    "magnification",
    "mean_value",
    "metadata",
    "model",
    "moving",
    "operator_mode",
    "optics",
    "outlier",
    "outlier_indices",
    "outlier_reason",
    "peak_value",
    "plane_result",
    "objective_magnification",
    "pixels",
    "points",
    "saturated",
    "serial_number",
    "stage",
    "valid",
    "width_source",
    "zscan_result",
    "z_scan",
}

DIMENSIONLESS_NAMES = {
    "camera_model",
    "objective_magnification",
    "raw_image_path",
    "processed_data_path",
    "pixels",
}


def has_allowed_unit_suffix(field_name: str) -> bool:
    """Return True for legacy contracts or known unit suffixes."""

    return field_name in DIMENSIONLESS_NAMES or field_name.endswith(UNIT_SUFFIXES)


def is_allowed_contract_field_name(field_name: str) -> bool:
    """Return True when a shared contract field name follows the unit convention.

    Physical quantities should end with a unit suffix. Logical, textual, enum,
    image-array, and explicitly dimensionless fields are allow-listed.
    """

    return (
        field_name in DIMENSIONLESS_FIELD_NAMES
        or field_name in DIMENSIONLESS_NAMES
        or field_name.endswith(UNIT_SUFFIXES)
    )
