# Changelog

## [v1.0.5] - 2026-04-28
- Refactored testfisico.py: removed duplicate plots, replaced phase‑space section with energy‑vs‑velocity analysis.
- Added Rodal (2025) irrotational metric with Hessian‑based proper energy density.
- Added Fuchs (2024) WarpShell audit (via warpax): zero negative energy across full subluminal parameter space.
- New convergence test and phase‑space sweep for Fuchs metric.
- Centralized constants in constants.py.
- Added pytest suite with Ford‑Roman bound regression test.
- Updated README with full metric comparison table.

## [v1.0.4] - 2026-04-28
- Added warpax‑powered Fuchs (2024) WarpShell audit.
- Convergence‑verified: E_minus = 0 at four grid resolutions.
- Full subluminal parameter sweep: zero negative energy across all tested (v/c, r_s_param) combinations.
- New phase‑space map (fuchs_phase_space.png).
- Rodal (2025) metric integrated with Hessian‑based proper energy density.
- Refactored test suite, centralized constants, pytest passing.

## [v1.0.3] - 2026-04-28
- Added numerical implementation of the Rodal (2025) kinematically irrotational warp drive.
- Proper energy density computed from spatial Hessian invariant; validated against canonical paper results.
- 38× peak deficit reduction vs. Alcubierre; net proper energy consistent with zero.
- QI gap now uses the actual diffuse negative‑energy volume.
- New output: `rodal_energy_map.png` polar density map.

## [v1.0.2] - 2026-04-28
- Added JOSS paper.md, paper.bib, example_figure.png, CITATION.cff, and GitHub Actions workflow for PDF compilation.

## [v1.0.1] - 2026-04-28
- Few adjustments.

## [v1.0.0] - 2026-04-27
- Initial public release. Quantum inequality audit for Alcubierre, White-Natário, and Lentz warp metrics. See README for details.
