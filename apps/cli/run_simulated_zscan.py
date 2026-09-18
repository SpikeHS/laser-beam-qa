"""Compatibility wrapper for the simulated z-scan CLI module."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from lbqa_cli.simulated_zscan import main, run_simulated_zscan  # noqa: E402

__all__ = ["main", "run_simulated_zscan"]


if __name__ == "__main__":
    raise SystemExit(main())
