---
title: 'warp-qi-audit: A Python toolkit for auditing warp drive metrics with quantum inequalities'
tags:
  - Python
  - general relativity
  - warp drive
  - quantum inequalities
  - numerical relativity
authors:
  - name: Mauricio Alejandro Bravo González
    orcid: '0000-0000-0000-0000'
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 2026-04-28
bibliography: paper.bib
---

# Summary

Warp drive metrics in general relativity promise superluminal travel through local spacetime distortion. However, their physical viability hinges on the ability to violate averaged energy conditions, which are constrained by quantum inequalities (QIs). `warp-qi-audit` is an open-source Python toolkit that numerically evaluates the Alcubierre, modified White-Natário, and Lentz metrics against the Ford–Roman quantum inequality bound. The code quantifies the degree of energy condition violation and checks whether the required negative-energy densities can be sustained within the limits set by quantum field theory.

# Statement of Need

Since Alcubierre's original 1994 metric, numerous modifications have been proposed aiming to reduce the energy requirements or eliminate the need for exotic matter. A key theoretical tool to vet these proposals is the quantum inequality (QI) derived by Ford and Roman, which provides a fundamental limit on the magnitude and duration of negative energy fluxes. Despite its importance, a ready-to-use, open-source implementation of QI constraints applied to different warp metrics has been lacking. `warp-qi-audit` fills this gap by providing:

- A modular framework to define arbitrary warp bubble shapes and stress-energy tensor components.
- Numerical evaluation of the Ford–Roman QI bound for a user-specified warp metric.
- Direct comparison with recent peer-reviewed results to validate the toolchain.
- Clear, annotated Jupyter notebooks for educational and reproducibility purposes.

The code has been verified against published results: it correctly reproduces the 68–69 orders of magnitude quantum inequality violation for the Alcubierre drive, the reduced peak energy density of the White-Natário modification, and the positive-energy requirement with numerical instability for the Lentz metric. These outcomes are consistent with independent peer-reviewed studies [@Lobo:2024; @Lentz:2021], confirming the software's reliability as an auditing tool.

# Usage and Implementation

The core computation relies on the `metric_explorer.py` module, which defines bubble profiles, calculates the relevant stress-energy components, and evaluates the QI bound via numerical integration. Dependencies include NumPy, SciPy, SymPy, and Matplotlib. Example notebooks (provided in the repository) step through each metric and visualize the energy density violations. The software is version-controlled on GitHub and permanently archived on Zenodo with DOI: 10.5281/zenodo.19862376.

# Figures

![Quantum inequality violation audit for Alcubierre, White-Natário, and Lentz metrics.](example_figure.png)

# Acknowledgements

We acknowledge the foundational work of Ford, Roman, and Pfenning on quantum inequalities, and the warp metric derivations by Alcubierre, White, Natário, and Lentz.

# References
