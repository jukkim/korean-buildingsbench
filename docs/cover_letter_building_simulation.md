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
Associate Professor
Department of Electrical Engineering
Sangmyung University
Seoul 03016, South Korea
E-mail: jukim@smu.ac.kr
ORCID: 0000-0001-9576-8757
