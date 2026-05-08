# Title Page

## Cover Letter

Dear Editor-in-Chief Prof. Da Yan,

I am submitting the enclosed manuscript entitled "Operational Diversity by Design: A Parametric Simulation Methodology for Zero-Shot Building Load Forecasting" for consideration as a Research Article in Building Simulation.

This study presents a parametric simulation design methodology for zero-shot building load forecasting. We address a fundamental question: can carefully designed EnergyPlus simulations, with operational schedules sampled through 12-dimensional Latin Hypercube Sampling, substitute for massive synthetic training corpora? The methodology is grounded in the Operational Diversity Hypothesis—that forecasting accuracy is governed by the coverage of distinct temporal load patterns, not by corpus scale.

The methodology defines a 12-dimensional operational schedule parameter space and uses LHS to produce training corpora that cover the space of plausible commercial building operations. We validate this approach through equal-scale controlled experiments, n-scaling analysis, cross-climate transfer tests, and comparison against the state-of-the-art 900,000-building benchmark (Emami et al., NeurIPS 2023). Just 700 LHS-designed buildings reach parity with the full benchmark on 955 real U.S. and Portuguese commercial buildings, and performance shows diminishing returns beyond 70 buildings—evidence that pattern coverage, not corpus scale, governs zero-shot generalization.

We believe this work is well-suited to Building Simulation because:

1. The core contribution is a simulation design methodology—how to construct parametric EnergyPlus simulations that maximize the diversity of temporal load patterns for machine learning applications.

2. The findings have direct practical implications for the simulation community: a single workstation day of simulation suffices where national building stock databases were previously assumed necessary.

3. The paper demonstrates that the design of the simulation parameter space matters more than simulation volume, a principle relevant to many building simulation applications beyond load forecasting.

The manuscript has not been published elsewhere and is not under consideration by another journal. The author has approved the manuscript and agrees with its submission to Building Simulation.

Thank you for considering this manuscript.

Sincerely,

Jeong-Uk Kim, Ph.D.

---

## Title

Operational Diversity by Design: A Parametric Simulation Methodology for Zero-Shot Building Load Forecasting

## Author

Jeong-Uk Kim

Department of Electrical Engineering, Sangmyung University, Seoul 03016, South Korea

E-mail: jukim@smu.ac.kr

ORCID: 0000-0001-9576-8757

---

## Author Contributions

Jeong-Uk Kim: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing — original draft, Writing — review & editing, Visualization.

## Compliance with Ethical Standards

### Declaration of competing interest

Financial interests: The author declares no financial interests. A Korean patent application covering the LHS-based simulation design methodology described in this paper was filed on May 1, 2026; its value may be affected by publication of this manuscript.

Non-financial interests: None.

### Ethical approval

This study does not contain any studies with human or animal subjects performed by the author.

## Acknowledgements

This work was supported by the Korea Institute of Energy Technology Evaluation and Planning (KETEP) grant funded by the Ministry of Trade, Industry and Energy, Republic of Korea (Grant No. RS-00238487).

During the preparation of this work the author used Claude (Anthropic) to assist with code development for simulation post-processing, figure generation scripts, and manuscript formatting. The author reviewed and edited all content and takes full responsibility for the content of the published article.

## Data Availability

The simulation pipeline, model source code, and evaluation scripts are publicly available at https://github.com/jukkim/korean-buildingsbench. The training data and trained model checkpoints can be regenerated from the provided pipeline or are available from the corresponding author upon reasonable request. The BuildingsBench evaluation data (BDG-2 and Electricity datasets) are available from the original BuildingsBench repository (https://data.openei.org/submissions/5859) under a CC-BY 4.0 license. EnergyPlus reference building IDFs are derived from DOE commercial prototype buildings, which are publicly available from the U.S. Department of Energy.
