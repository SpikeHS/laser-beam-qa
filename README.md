# Laser Beam QA

[![CI](https://github.com/SpikeHS/laser-beam-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/SpikeHS/laser-beam-qa/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Hardware-free laser beam analysis, simulated z-scans and inspectable reports.**

Developed from laser-characterization work in the N09 laboratory at the
Institute of Physics, Chinese Academy of Sciences. This first public version
extracts the reusable analysis and simulation core of a larger local project.
It can be used and tested without a camera, motor stage or vendor SDK.

[中文说明](README.zh-CN.md) · [Methods](docs/methods.md) · [Roadmap](ROADMAP.md)

## What it does

- Measures intensity-weighted centroid, D4σ, FWHM and ellipse geometry.
- Simulates an astigmatic beam, image acquisition and a guarded z-scan sequence.
- Fits a Gaussian-caustic z-scan; provides a separate far-field linear-fit API.
- Saves calibration/recipe snapshots, CSV/JSON results and HTML reports.
- Includes deterministic synthetic images, numerical regressions and end-to-end tests.

## Quick start

Python 3.11 or newer:

```sh
git clone https://github.com/SpikeHS/laser-beam-qa.git
cd laser-beam-qa
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
lbqa-simulated-zscan --recipe configs/recipe_40x_zscan.example.yaml --calibration configs/calibration_40x.example.yaml --sample-id SIM001 --out runs
```

Open the generated `runs/Run_.../report.html`. The run also includes
`z_scan_table.csv`, `result_summary.csv`, `result_summary.json` and configuration
snapshots. The simulated truth is 8 mrad / 10 mrad full-angle divergence for
the X/Y axes. These are example parameters, not laboratory measurements.

```sh
python -m pytest -q
python -m build
```

The CLI is also available as `python -m apps.cli.run_simulated_zscan` from a
source checkout. A built wheel installs the `lbqa-simulated-zscan` command;
use the example configuration files from the checkout or source archive.

## Scope and scientific interpretation

Physical fields include units, such as `z_actual_mm` and `full_angle_x_mrad`.
Nominal example calibration is not a calibration of your equipment. The z-scan
CLI uses a Gaussian-caustic fit; the separate far-field linear fit does not
infer a beam waist. See [method definitions and limitations](docs/methods.md).

Successful scans keep final numeric results, configuration snapshots and report
images; raw grayscale frames are temporary report inputs and are not retained.
For reproducibility, this repository provides synthetic generators and fixtures.
Real-measurement archiving is a future design item.

The first public version does not include the local desktop UI, real-device
adapters, factory software or measurement datasets. It makes no claim of a
validated autonomous experiment system or universal beam-quality certification.

## Related research tools

- [PL Analyzer](https://github.com/SpikeHS/PL-Analyzer): photoluminescence analysis.
- [Laser Characterization Tools](https://github.com/SpikeHS/laser-characterization-tools): spectral mapping and LIV analysis.

The longer-term goal is traceable material/device measurements that can support
predictive-model development as suitable datasets accumulate. Cross-project
sample linking and predictive models remain planned work.

## License and contribution

MIT © 2026 Sen Hu. See [attribution](ATTRIBUTION.md) and
[contributing](CONTRIBUTING.md). Please report incorrect results with a small,
shareable example and explicit units.
