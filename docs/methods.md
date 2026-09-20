# Methods and reproducibility

Laser Beam QA provides optical quality testing and analysis algorithms for
laser-beam intensity and propagation measurements. The analysis functions are
independent of the simulated acquisition workflow; calibrated image dimensions,
axial positions and an appropriate model remain the caller's responsibility.

- **D4σ:** four times the intensity-weighted standard deviation of the
  background-corrected image, with separate image axes and principal axes.
  Background, clipping, ROI and saturation affect second moments.
- **FWHM:** image-profile half-maximum widths; this is not interchangeable with
  D4σ for arbitrary beam shapes.
- **Gaussian-caustic z-scan:** the simulated CLI fits diameter versus axial
  position through the Gaussian-caustic model. Synthetic tests check known
  divergence, displaced waists and validity behavior.
- **Far-field fit:** a separate linear diameter-versus-z fit, with independently
  selected axes/points and residual reporting. A line does not determine a
  physical waist. Two points determine a line but not residual scatter.

Full-angle and half-angle divergence have separate unit-bearing fields. Tests
cover formulas, unit contracts, numerical examples, simulation and reports;
they do not constitute calibration or experimental validation for every beam.

The `tests/golden_images/synthetic_zscan_40x` TIFFs are generated fixtures, not
measurements. Their expected-results JSON records synthetic parameters and
reference outputs. Additional generators live in `lbqa_analysis.synthetic`
and `lbqa_devices.simulated.synthetic_beam`.

Final runs retain numerical outputs, calibration/recipe snapshots and rendered
spot images. Temporary grayscale sources are cleaned up on successful report
generation. Keep independent raw inputs if adapting these APIs to real data.
