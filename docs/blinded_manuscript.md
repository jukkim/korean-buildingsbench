# Operational Diversity by Design: A Parametric Simulation Methodology for Zero-Shot Building Load Forecasting

---

## Abstract

Zero-shot building load forecasting requires simulation datasets that capture diverse real-world operational patterns. Existing approaches often rely on large building-stock databases, but less attention has been paid to how simulation data should be designed to cover temporal operating regimes. This paper proposes a parametric EnergyPlus simulation design methodology grounded in the Operational Diversity Hypothesis: zero-shot generalization depends not only on the number of simulated buildings, but also on the coverage of distinct temporal load patterns.

The proposed methodology parameterizes commercial building operation across key schedule dimensions, including operating hours, baseload behavior, equipment retention, weekly disruptions, seasonal variation, and stochastic noise. Latin Hypercube Sampling is used to construct a compact simulation dataset that deliberately spans this operational schedule space across commercial building archetypes. The resulting simulations are combined with Reversible Instance Normalization and data augmentation to train a Transformer-based zero-shot load forecasting model.

Evaluation on real commercial building loads shows that the designed simulation dataset achieves benchmark-level transfer performance while requiring far fewer simulations than large stock-model pipelines. Equal-scale controls, weather-origin ablations, and dataset-size scaling experiments indicate that systematic coverage of operational patterns is a major contributor to transferability. These findings support a data-centric simulation design approach for zero-shot building load forecasting, shifting the data requirement from brute-force stock-model scaling toward targeted operational diversity.

**Keywords**: operational diversity, zero-shot forecasting, building energy, foundation models, data-centric AI, Latin Hypercube Sampling

---

## 1. Introduction

### 1.1 Simulation Design for Zero-Shot Building Load Forecasting

Short-term load forecasting underpins grid balancing, demand response, and real-time building energy management (Fan et al. 2017; Hong et al. 2019). Foundation models for time series—trained once and transferred to unseen targets—promise to replace the classical per-building approach, but their success depends critically on how the training corpus is designed. This design question has received surprisingly little attention: existing approaches have demonstrated the value of large-scale building stock databases, but the design of simulation data for temporal pattern coverage remains underexplored.

Large stock-model pipelines provide an important reference point for zero-shot building load forecasting. For example, the Buildings-900K corpus (Wilson et al. 2022; Emami et al. 2023) draws from the NREL End-Use Load Profiles database and serves as a widely used external reference for zero-shot building load forecasting. However, such pipelines are primarily designed to represent building-stock distributions, not to systematically span the operational schedule space. Buildings drawn from the same stock-model generation pipeline share common operational assumptions—similar occupancy schedules, equipment profiles, and HVAC operating logic—so adding more buildings may increase sample size without proportionally expanding the range of temporal dynamics the model encounters.

Recent work on time series scaling laws suggests this distinction matters. Shi et al. (2024) showed that scaling behavior in time series forecasting diverges from language modeling: performance depends on the interaction between look-back horizon, autocorrelation, and non-stationarity, not simply on data volume. MOIRAI-MoE (Liu et al. 2025) achieved superior zero-shot performance through architectural specialization rather than increased scale. These findings motivate a systematic investigation of what properties of the training data actually govern zero-shot generalization in building energy forecasting.

### 1.2 The Operational Diversity Hypothesis

This paper proposes a simulation design methodology grounded in a specific theoretical claim: the **Operational Diversity Hypothesis**. In zero-shot building load forecasting, prediction accuracy depends not only on the number of training buildings, but also on the diversity of temporal load patterns in the training set—the coverage of the operational schedule space.

The hypothesis is motivated by a structural property of building energy time series. The temporal dynamics of a building's load profile are shaped primarily by its operational schedule—operating hours, baseload behavior, equipment retention, weekly patterns, and seasonal variation. Stock-model corpora draw from statistical distributions of real building stocks, meaning most generated buildings cluster around typical operating conditions: offices running 08:00–18:00, hospitals operating continuously, retail with moderate baseloads. Increasing the number of buildings from such a distribution adds more samples near the mode without necessarily expanding the range of temporal patterns.

If temporal pattern coverage—not sample count—is the operative factor, then an alternative design strategy becomes possible: construct a small training set that broadly covers the operational schedule space through systematic sampling. A carefully designed dataset of hundreds of buildings may therefore achieve comparable transfer performance to much larger stock-model datasets.

### 1.3 Methodology and Contributions

We instantiate this hypothesis through a concrete simulation design methodology. We generate 700 EnergyPlus simulations (50 per archetype × 14 commercial building types) with operational schedules sampled via 12-dimensional Latin Hypercube Sampling (LHS). The 12 parameters span operating hours, baseload characteristics, equipment retention, weekly disruptions, seasonal variation, and stochastic noise—designed to cover the space of plausible commercial building operations. Combined with Reversible Instance Normalization (RevIN) (Kim et al. 2022) and data augmentation, this corpus trains a standard Transformer with Gaussian NLL loss.

Both LHS and RevIN are established techniques; the contribution is demonstrating that their systematic combination reframes the data requirement for zero-shot building energy forecasting from a scale problem to a design problem. The contributions are:

1. **A parametric simulation design methodology for zero-shot building load forecasting.** We define a multi-dimensional operational schedule parameter space and use LHS to promote broad coverage of temporal operating patterns across commercial building archetypes. The methodology produces a compact training dataset from EnergyPlus simulations, reducing the data-engineering effort associated with assembling national-scale stock-model datasets.

2. **A data-centric validation framework for simulation design.** We evaluate the designed simulation dataset under matched training conditions, weather-origin ablations, and dataset-size scaling experiments. These controlled comparisons examine whether operational schedule coverage improves zero-shot transferability independently of confounding factors. We use the BuildingsBench evaluation protocol (Emami et al. 2023) as an external reference for zero-shot transfer performance, but the focus of this work is the design of the simulation dataset rather than benchmark competition.

3. **A normalization analysis for designed simulation datasets.** RevIN interacts with the information structure of the training data: it helps when load magnitude is decoupled from schedule dynamics and degrades performance when magnitude carries stock-model-specific information. This regime-dependent effect has implications for the broader time series community.

4. **Practical guidance for compact simulation dataset construction.** The proposed workflow demonstrates that targeted operational diversity can achieve benchmark-level zero-shot transfer, reducing the need for national-scale stock-model pipelines. Weather-origin ablations confirm that the benefit persists across different climate files.

---

## 2. Related Work

### 2.1 Building Energy Forecasting and Zero-Shot Transfer

Building energy forecasting has progressed from statistical models (Amasyali and El-Gohary 2018) through gradient boosting (Chen and Guestrin 2016), recurrent networks (Hochreiter and Schmidhuber 1997), and Transformers (Vaswani et al. 2017; Nie et al. 2023). These methods achieve high per-building accuracy but require target-building training data. Transfer learning through domain adaptation (Ribeiro et al. 2018) and fine-tuning (Spencer et al. 2025) relaxes this constraint but still needs target-domain samples.

The zero-shot paradigm eliminates target-building data entirely: a model trained on synthetic simulations generalizes to unseen real buildings at inference time. Emami et al. (2023) operationalized this through a 900K-building corpus drawn from the NREL EULP pipeline (Wilson et al. 2022), training a Transformer with Gaussian NLL loss for approximately 0.067 epochs—roughly 2–3 sampled windows per building—using global Box-Cox normalization (λ = −0.067). The sub-epoch training regime is consequential: additional training degraded out-of-distribution performance, which the authors interpreted as evidence that the model memorizes synthetic patterns faster than it learns transferable representations. This observation is consistent with the Operational Diversity Hypothesis—the useful information may be learned quickly because many buildings produce similar temporal patterns.

### 2.2 Scaling Laws and Data Efficiency in Time Series

The proliferation of general-purpose time series foundation models—TimesFM (Das et al. 2024), Chronos (Ansari et al. 2024), Lag-Llama (Rasul et al. 2023), MOIRAI (Woo et al. 2024)—has established that large-scale pretraining enables zero-shot forecasting across domains. Yao et al. (2025) found that model architecture significantly influences scaling efficiency. Most relevant to our work, Shi et al. (2024) showed that in time series, additional data does not expand the combinatorial coverage of contexts as directly as additional tokens do in language modeling. Our findings provide a concrete, domain-specific instance of this principle: in building energy, the marginal information content of additional stock-model buildings decreases rapidly because they occupy similar regions of operational schedule space.

### 2.3 Data-Centric AI and Instance Normalization

The data-centric AI perspective (Zha et al. 2023) holds that improving data quality is often more productive than improving model architecture. Our 12-dimensional LHS design applies this principle to the energy domain: rather than sampling from a fixed distribution, we construct a training set that fills the space of plausible operations.

RevIN (Kim et al. 2022) normalizes instance-level load magnitude and variability before the encoder and restores the scale after prediction. Our finding that RevIN's benefit is regime-dependent—helpful for small diverse datasets, harmful at large scale—extends the understanding of when instance normalization helps versus hurts, connecting to broader questions about the role of magnitude information in time series models.

---

## 3. Method

### 3.1 Parametric Building Simulation with LHS-Designed Operational Diversity

Training data are generated through EnergyPlus (Crawley et al. 2001) simulation of commercial buildings with parametrically varied operational schedules. The goal is not to replicate a building stock distribution but to promote broad coverage of the temporal pattern space.

**Building archetypes.** We use 14 building archetypes based on DOE commercial reference building models (Deru et al. 2011) adapted to Korean building codes and climate zones: office, retail, school, hotel, hospital, apartment (midrise and highrise), small office, large office, warehouse, strip mall, restaurant (full-service and quick-service), and university. Each archetype begins from a DOE reference model with code-compliant envelope thermal properties, HVAC system sizing, and internal load densities.

**Climate coverage.** Simulations span five Korean cities across three climate zones: Seoul (central), Busan (southern), Daegu (southern), Gangneung (central-coastal), and Jeju (subtropical). For the U.S.-TMY ablation (Section 4.2), the same 700 schedule samples and archetypes were rerun with U.S. TMY files mapped by approximate ASHRAE climate-zone similarity: Seoul → Washington DC (4A), Busan → Atlanta (3A), Daegu → Charlotte (3A), Gangneung → Boston (5A), and Jeju → Miami (1A).

**Operational parameter space.** Building operational schedules are parameterized by 12 continuous variables sampled via Latin Hypercube Sampling (McKay et al. 1979) (Table 1). Unlike stock-model sampling—which reflects the statistical distribution of real buildings (many near typical conditions, few at extremes)—LHS ensures uniform marginal coverage across all dimensions. Stock-model sampling asks "what does the building stock look like?"; LHS asks "what temporal patterns can buildings produce?"

**Table 1.** Twelve-dimensional LHS parameter space for operational schedule generation.

| Parameter | Range | Description |
|-----------|-------|-------------|
| op_start | 0–12 h | Operation start time |
| op_duration | 8–24 h | Operating hours per day |
| baseload_pct | 25–98% | Off-hours load as fraction of peak |
| weekend_factor | 0–1.2 | Weekend-to-weekday load ratio |
| ramp_hours | 0.5–4 h | Transition ramp duration |
| equip_always_on | 30–95% | Always-on equipment fraction |
| daily_noise_std | 5–35% | Day-to-day stochastic variation |
| scale_mult | 0.3–3.0 | Load density multiplier |
| night_equipment_frac | 30–95% | Nighttime equipment retention |
| weekly_break_prob | 0–25% | Weekly pattern disruption probability |
| seasonal_amplitude | 0–30% | Seasonal load oscillation amplitude |
| process_load_frac | 0–50% | Constant process load fraction |

The resulting training set ranges from 24/7 high-baseload facilities (resembling data centers or hospitals) to weekday-only offices with steep morning ramps and low overnight loads. For the n = 50 configuration used in our main experiments, this yields 50 × 14 = 700 buildings. Each sample modifies the EnergyPlus IDF file, runs a full annual simulation (8,760 hours), and extracts hourly total electricity consumption. The end-to-end pipeline—from archetype selection through LHS sampling, simulation, normalization, and zero-shot inference—is illustrated in Fig. 1.

**Post-hoc distributional validation.** Table 2 provides a post-hoc comparison showing that including the four diversity parameters (night_equipment_frac, weekly_break_prob, seasonal_amplitude, process_load_frac) moves several key profile statistics toward the range observed in real buildings, although not uniformly across all metrics. These statistics were computed after defining the parameter space and were not used for model selection or hyperparameter tuning. The overshoot in baseload P95 and CV P5 reflects the LHS design's intent to cover extreme high-baseload and low-variability regimes rather than to match the empirical stock distribution.

**Table 2.** Post-hoc comparison of simulated and real-building load-profile statistics. Metrics characterize input sequence shape (X), not prediction targets (y).

| Metric | Without 4 params | With 4 params | Real Buildings* |
|--------|:----------------:|:-------------:|:-----------------:|
| Night/Day ratio | 0.512 | 0.852 | 0.803 |
| Autocorrelation (168h) | 0.938 | 0.784 | 0.751 |
| Baseload P95 | 0.51 | 0.905 | 0.725 |
| CV P5 (flatness) | 0.21 | 0.032 | 0.105 |

*Real Buildings = the 955-building evaluation set from the BuildingsBench benchmark (Emami et al. 2023).

### 3.2 Model Architecture

We use a standard encoder-decoder Transformer: 3 encoder and 3 decoder layers, d_model = 512, 8 attention heads, feedforward dimension 1024, totaling 15.8M parameters (Transformer-M). Input consists of 168 hourly load values with temporal features (day-of-year sinusoidal encoding, day-of-week and hour-of-day embeddings). The model predicts 24-hour-ahead Gaussian distributions through autoregressive decoding with Gaussian NLL loss. Using the same Transformer backbone as Emami et al. (2023) minimizes model-capacity confounds in the controlled comparisons.

**Reversible Instance Normalization.** Before encoding, the 168-hour context window is normalized by its instance mean and standard deviation; after decoding, the normalization is reversed. This removes building-specific load magnitude and variability, allowing the model to focus on temporal shape. We apply RevIN symmetrically in all controlled experiments, including stock-model baselines, to ensure fair comparison. The comparison with the original stock-model pipeline (which does not use RevIN) is discussed in Section 5.1.

**No geographic features.** Some zero-shot approaches provide latitude and longitude embeddings as model inputs. We set both to zero for all buildings—both training and evaluation—to test whether operational diversity alone suffices for generalization. The ablation in Section 4.4 confirms that actual coordinates provide no benefit, consistent with RevIN absorbing the scale differences that coordinates might otherwise encode.

### 3.3 Training Protocol

We apply a global Box-Cox transform (Box and Cox 1964) fitted on our simulation data (λ = −0.067). AdamW (Loshchilov and Hutter 2019) with learning rate 6 × 10⁻⁵, weight decay 0.01, cosine annealing, and 500-step warmup. Training runs for 18,000 gradient steps with batch size 128, corresponding to approximately 9 epochs over 700 buildings. Data augmentation includes window jitter (±1–6h), Gaussian noise (σ = 0.02 in Box-Cox space), and amplitude scaling (U[0.85, 1.15]).

With 700 buildings and 18,000 steps, each building is seen roughly 3,300 times across window positions—in contrast to the stock-model baseline, where each of 900K buildings contributes roughly 2–3 windows. Three factors mitigate overfitting despite this repeated exposure: (1) augmentation ensures each exposure presents a different view; (2) RevIN prevents memorizing absolute load levels; (3) the high inter-building diversity from LHS ensures patterns learned from any one building generalize.

### 3.4 Evaluation Protocol

We evaluate on a standardized real-building test set of 955 commercial load series (Emami et al. 2023): 611 from U.S. commercial buildings in the Building Data Genome Project 2 (BDG-2) (Miller et al. 2020), drawn from four university and government campuses spanning Mediterranean, semi-arid, humid subtropical, and humid continental climates, and 344 from Portuguese electricity consumers (Trindade 2015), with 15 out-of-vocabulary buildings excluded per the original specification. Per-building NRMSE = sqrt(MSE) / mean(actual), aggregated by median across buildings. The sliding window uses stride 24 hours, context 168 hours, and prediction horizon 24 hours. We reproduced the stock-model baseline using the official checkpoint and obtained 13.27%, confirming pipeline equivalence within 0.01 pp of the reported 13.28%.

---

## 4. Experiments and Results

### 4.1 Equal-Scale Controlled Validation of Simulation Design

The central experiment tests whether the proposed simulation design methodology produces more transferable training data than stock-model sampling. Two 700-building datasets are compared under matched training conditions, with simulation data design as the primary variable of interest (Table 3, Fig. 2).

**Table 3.** External zero-shot validation on real commercial buildings (955-building evaluation set). Korean-700 and US-TMY-700 report five-seed mean ± std; Korean-700 (no aug) and Korean-700 (RevIN OFF) report three-seed mean ± std; all other rows use seed = 42. †BB-900K Normalized Continuous Ranked Probability Score (NCRPS) unavailable due to Box-Cox protocol mismatch. BB-700 (aug-matched) uses identical optimizer, augmentation, and steps as Korean-700. BB denotes the BuildingsBench stock-model corpus (Emami et al. 2023).

| Model | Role | Data | N | RevIN | Aug | NRMSE (%) | NCRPS (%) | Δ baseline |
|-------|------|------|------:|:-----:|:---:|:---------:|:---------:|:------:|
| **K-700 (ours)** | **Proposed** | **LHS sim** | **700** | **ON** | **ON** | **13.11 ± 0.17** | **7.14 ± 0.03** | **−0.16** |
| **US-700 (ours)** | **Weather ablation** | **US TMY sim** | **700** | **ON** | **ON** | **13.64 ± 0.65** | **7.53 ± 0.41** | **+0.37** |
| K-700 no aug | Aug ablation | LHS sim | 700 | ON | OFF | 13.67 ± 0.21 | 8.16 | +0.40 |
| K-700 no RevIN | RevIN ablation | LHS sim | 700 | OFF | ON | 14.72 ± 0.28 | 8.29 | +1.45 |
| BB-700 aug | Equal-size control | BB subset | 700 | ON | ON | 14.26 | 7.80 | +0.99 |
| BB-700 | Control (no aug) | BB subset | 700 | ON | OFF | 15.28 | — | +2.01 |
| BB-900K | External reference | BB 900K | 900,000 | OFF | OFF | 13.27 | —† | — |
| BB+RevIN | Ref. + RevIN | BB 900K | 900,000 | ON | OFF | 13.89 | 7.76 | +0.62 |
| Persist. | Naive baseline | — | — | — | — | 16.68 | — | +3.41 |

The equal-scale comparison provides evidence for the central claim of the proposed methodology: that deliberate design of operational diversity can produce more transferable training data than sampling from a fixed stock-model distribution. Under matched conditions (same architecture, same optimizer, same augmentation, same number of buildings, same RevIN), the LHS-designed corpus outperforms the stock-model sample by **1.33 pp** at seed 42 (12.93% vs. 14.26%). Although the stock-model control (BB-700 aug) was not evaluated across multiple seeds, the 1.33 pp gap is substantially larger than the observed inter-seed variation of Korean-700 (±0.17 pp), suggesting that the advantage is unlikely to be explained solely by seed noise. Even without augmentation, Korean-700 (13.67%) outperforms the aug-matched stock-model control by 0.59 pp, and the U.S.-weather variant (13.64%) by 0.62 pp—smaller but directionally consistent advantages. Because RevIN, augmentation, model architecture, and training budget are matched, the gap is unlikely to be explained by these factors. The primary remaining difference is the design and source of the training data, although uncontrolled factors such as building codes, envelope properties, and climate mapping cannot be fully excluded (Section 5.4).

As a secondary observation, the designed dataset achieves performance comparable to the large stock-model reference (five-seed mean 13.11 ± 0.17% vs. 13.27%) without geographic metadata. A paired per-building comparison shows Korean-700 achieves lower error on 680 of 955 buildings (71%); the paired bootstrap 95% CI of the median per-building NRMSE difference is [0.31, 0.39] pp (Wilcoxon signed-rank p < 0.001). This per-building paired effect is larger than the 0.16 pp gap between aggregate medians because paired differencing removes inter-building variance.

### 4.2 Weather-Origin Ablation

To test whether the observed benefit depends on the Korean weather files, we retrained on the same 700 LHS-designed buildings using U.S. TMY weather (Seoul → Washington DC, Busan → Atlanta, Daegu → Charlotte, Gangneung → Boston, Jeju → Miami). The five-seed mean of US-TMY-700 is 13.64 ± 0.65%, within 0.53 pp of Korean-700. The U.S.-TMY model still outperforms the equal-scale stock-model control (14.26%) by 0.62 pp, suggesting that schedule design remains beneficial after replacing the Korean weather files with approximately matched U.S. TMY files. The higher seed variance (0.65% vs. 0.17%) indicates that weather choice affects run-to-run stability.

### 4.3 N-Scaling: Evidence for Pattern-Count-Governed Learning

If the Operational Diversity Hypothesis is correct, performance should show diminishing returns once the LHS parameter space is adequately covered. Table 4 and Fig. 3 test this prediction.

**Table 4.** N-scaling results (Transformer-M, RevIN ON, aug ON, s = 18,000, seed = 42). Exception: n = 5 reports five-seed mean ± std.

| n | Total Buildings | NRMSE (%) | Δ baseline |
|:-:|:---------------:|:----------:|:-------:|
| 1 | 14 | 14.72 | +1.45 |
| 3 | 42 | 13.47 | +0.20 |
| 5 | 70 | 13.28 ± 0.12 | +0.01 |
| 10 | 140 | 13.18 | −0.09 |
| 20 | 280 | 13.23 | −0.04 |
| 50 | 700 | 12.93 | −0.34 |
| 70 | 980 | 13.20 | −0.07 |
| 80 | 1,120 | 13.15 | −0.12 |

Under the fixed 18,000-step training budget, performance improves sharply from n = 1 to n = 5 (14.72% to 13.28%), plateauing near the external reference level (13.27%) from n ≈ 5 onward. Beyond this point, gains become incremental—fluctuations within ±0.35 pp rather than clear monotonic improvement—consistent with the interpretation that the designed parameter space has been adequately covered. The best single-seed result (12.93% at n = 50) falls within this plateau, suggesting that additional samples refine existing patterns rather than introducing qualitatively new ones.

The contrast with stock-model scaling is instructive (Appendix B): increasing from 700 to 7,000 stock-model buildings improves NRMSE by only 0.78 pp (15.28% to 14.50%), and even 7,000 stock-model buildings do not approach 700 LHS-designed buildings (13.11%). This is consistent with the interpretation that additional stock-model buildings occupy similar regions of operational space.

### 4.4 Ablation Studies

Table 5 reports ablations for RevIN and geographic metadata, using seed-42 for consistent comparison.

**Table 5.** Ablation results (seed 42, val_best checkpoint, 955 buildings).

| Experiment | NRMSE (%) | Delta | Interpretation |
|------------|:----------:|:-----:|----------------|
| K-700 RevIN ON (baseline) | 12.93 | — | Seed 42, lat/lon = zero |
| K-700 RevIN OFF | 14.81 | +1.88 | RevIN contributes 1.88 pp (seed 42); 1.6 pp across multi-seed means (Appendix A) |
| Actual lat/lon coordinates | 12.93 | 0.00 | Geographic metadata provides no benefit |

Additional ablations (Appendix C) show that using stock-model-fitted Box-Cox parameters (+3.31 pp), extended training (+3.09 pp), scaling to 70K buildings with fixed budget (+2.42 pp), and seasonal decomposition before RevIN (+3.72 pp) all degrade performance, each illustrating a distinct failure mode.

---

## 5. Discussion

### 5.1 Mechanistic Interpretation of RevIN's Regime-Dependent Effect

The asymmetry in RevIN's effect across dataset scales (Fig. 4) admits a coherent mechanistic interpretation that extends beyond our specific setting.

RevIN normalizes each context window to zero mean and unit variance, stripping absolute load magnitude. With a small LHS-designed dataset, the model has not seen enough magnitude variation to internalize it; RevIN reduces this burden by normalizing instance-level scale and variability, yielding a substantial improvement (Table 5). With a large stock-model corpus—where each building contributes only a few training windows—magnitude may carry transferable information because it is correlated with building type, HVAC configuration, and schedule assumptions in the generation pipeline. Removing this information may discard something useful, producing a measurable degradation (Table 3).

A structural factor reinforces this. In stock-model corpora, magnitude is informative about building type because HVAC, envelope, and schedule are jointly determined by the generation pipeline—magnitude and temporal shape are correlated. In our LHS design, magnitude reflects only the scale_mult parameter, sampled independently of schedule parameters. RevIN's removal of magnitude therefore discards less useful information in LHS data than in stock-model data.

This finding has implications for the broader time series community: RevIN's benefit depends not just on dataset size but on the information structure of the training data—specifically, whether magnitude carries predictive information about temporal dynamics.

### 5.2 Design Principles Behind Transferable Pattern Coverage

The methodology's effectiveness can be traced to three design principles that together produce a training corpus with high transferability per building.

1. **Independent marginal coverage through LHS.** Each of the 12 schedule parameters is sampled with uniform marginal coverage, ensuring that the training set visits the full range of each operational dimension—including extreme combinations (e.g., high baseload with short operating hours) that rarely occur in real building stocks but expose the model to temporal dynamics it must handle at inference time. This helps explain why 700 LHS-designed buildings outperform 700 stock-model buildings under identical conditions: LHS spreads samples across the operational schedule space rather than concentrating them around its mode.

2. **Archetype-stratified diversity.** Stratifying the 12D LHS across 14 building archetypes ensures that schedule diversity interacts with distinct thermal mass, HVAC response, and internal gain profiles. This produces a combinatorial expansion of temporal patterns: the same nighttime equipment retention fraction yields qualitatively different load shapes in a hospital versus a strip mall. The n-scaling plateau at a modest number of samples per archetype (Table 4) indicates that this archetype × schedule interaction space is covered with relatively few samples.

3. **Decoupled magnitude and temporal shape.** The scale_mult parameter is sampled independently of the 11 schedule parameters, reducing the extent to which load magnitude encodes schedule-driven temporal dynamics. This structural property helps explain why RevIN—which strips magnitude—helps in our design but hurts in stock-model corpora, where magnitude and schedule are jointly determined by the generation pipeline and therefore correlated.

These principles also explain why the sub-epoch training regime observed in large-scale stock-model training (Emami et al. 2023) is consistent with our hypothesis: the distinct temporal patterns in a stock-model corpus are learned quickly because many buildings produce similar patterns, and further training memorizes pipeline-specific artifacts rather than learning new dynamics.

From a scaling-law perspective, this aligns with Shi et al. (2024): in building load time series, the marginal information content of additional buildings decreases rapidly when the operational diversity of the corpus is fixed. Our methodology addresses this directly by promoting broad diversity at design time.

### 5.3 Practical Implications

The reframing from scale to design has immediate practical consequences. An organization can generate several hundred parametric EnergyPlus simulations conforming to local building codes and climate, train a standard Transformer with RevIN and augmentation, and deploy zero-shot forecasting—without assembling a national building stock database.

The computational cost is modest: in our implementation, 700 simulations completed in approximately 4 hours on a single workstation (10-core CPU, 64 GB RAM); training takes approximately 2 hours on a single consumer GPU (NVIDIA RTX 4090). The entire pipeline from simulation to deployed model takes less than a day. No location metadata, building-type classification, or HVAC information is required at inference time—only 168 hours of historical load data.

### 5.4 Limitations

Our parametric simulations differ from the stock-model baseline in schedule design, climate, building codes, and envelope parameters. The equal-scale controlled experiment (Section 4.1) controls for augmentation, model architecture, and training budget, but cannot fully disentangle all data-source-related confounds. The U.S.-TMY ablation substantially reduces but does not eliminate the climate-origin confound.

The augmentation applied to our method (window jitter, Gaussian noise, amplitude scaling) was not part of the original stock-model training pipeline. The aug-matched control addresses this asymmetry at equal scale, but the comparison with the original baseline (which uses no augmentation) includes this confound. Even without augmentation, our method outperforms the aug-matched stock-model control (Table 3), but falls above the full-scale baseline.

The objective of this study is not to claim a large performance breakthrough, but to demonstrate that a systematically designed simulation dataset can provide competitive zero-shot transfer with substantially lower data requirements. The margin between the designed dataset and the large-scale reference is comparable to inter-seed variation (Table 3). The paired per-building comparison confirms statistical significance (Section 4.1), but the margin is small. The contribution is methodological: showing that systematic simulation design can substantially reduce reliance on brute-force dataset scaling in this benchmark setting.

All experiments use a single model architecture (Transformer-M); generalization to PatchTST, MOIRAI, or other architectures has not been tested. On residential buildings, our model yields accuracy comparable to the Persistence Ensemble, confirming that zero-shot residential forecasting remains an open problem. Real-world validation beyond the evaluation set used in this study has not been conducted; broader testing across diverse building types in multiple countries is needed.

---

## 6. Conclusion

We presented a parametric simulation design methodology for zero-shot building load forecasting, grounded in the Operational Diversity Hypothesis. The methodology defines a multi-dimensional operational schedule parameter space, uses LHS to promote broad coverage of temporal operating patterns across commercial building archetypes, and combines the resulting dataset with RevIN and data augmentation. Four lines of evidence support the approach: (1) equal-scale controlled experiments provide evidence of a consistent data-design advantage across multiple comparisons; (2) n-scaling shows diminishing returns beyond a modest number of buildings, consistent with pattern-coverage-governed learning; (3) RevIN's regime-dependent effect is mechanistically consistent with the decoupled magnitude structure of LHS-designed data; and (4) cross-climate transfer shows that the benefit persists under different weather files, reducing the likelihood that results are attributable solely to climate origin.

The methodology reframes the data requirement for zero-shot building energy forecasting: from assembling a national building stock database to designing a few hundred simulations with broad operational diversity. This reduces the barrier to deployment in data-sparse regions from a data-engineering problem requiring access to national stock models to a simulation design problem solvable on a single workstation in less than a day.

The principal open questions are the limitation to commercial buildings, the augmentation asymmetry between training pipelines, and the untested generalization to other model architectures. Extending the operational diversity framework to residential buildings—where zero-shot performance remains at Persistence-level—is a natural next step.

---

## References

Amasyali K, El-Gohary NM (2018). A review of data-driven building energy consumption prediction studies. *Renewable and Sustainable Energy Reviews*, 81: 1192–1205.

Ansari AF, et al. (2024). Chronos: Learning the Language of Time Series. *Transactions on Machine Learning Research*.

Box GEP, Cox DR (1964). An Analysis of Transformations. *Journal of the Royal Statistical Society, Series B*, 26(2): 211–252.

Chen T, Guestrin C (2016). XGBoost: A Scalable Tree Boosting System. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794.

Crawley DB, et al. (2001). EnergyPlus: creating a new-generation building energy simulation program. *Energy and Buildings*, 33(4): 319–331.

Das A, et al. (2024). A decoder-only foundation model for time-series forecasting. *Proceedings of Machine Learning Research*, 235: 10148–10167.

Deru M, et al. (2011). U.S. Department of Energy Commercial Reference Building Models of the National Building Stock. Golden, CO: National Renewable Energy Laboratory. Report No.: NREL/TP-5500-46861.

Emami P, Sahu A, Graf P (2023). BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting. *Advances in Neural Information Processing Systems*, 36.

Fan C, et al. (2017). A short-term building cooling load prediction method using deep learning algorithms. *Applied Energy*, 195: 222–233.

Hochreiter S, Schmidhuber J (1997). Long Short-Term Memory. *Neural Computation*, 9(8): 1735–1780.

Hong T, et al. (2019). Ten questions on urban building energy modeling. *Building and Environment*, 168: 106508.

Kim T, et al. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. In: International Conference on Learning Representations.

Liu X, et al. (2025). MOIRAI-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts. *Proceedings of Machine Learning Research*, 267: 38940–38962.

Loshchilov I, Hutter F (2019). Decoupled Weight Decay Regularization. In: International Conference on Learning Representations.

McKay MD, Beckman RJ, Conover WJ (1979). A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. *Technometrics*, 21(2): 239–245.

Miller C, et al. (2020). The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data*, 7: 368.

Nie Y, et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In: International Conference on Learning Representations.

Rasul K, et al. (2023). Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting. In: NeurIPS Workshop on Robustness of Foundation Models.

Ribeiro M, et al. (2018). Transfer learning with seasonal and trend adjustment for cross-building energy forecasting. *Energy and Buildings*, 165: 352–363.

Shi J, et al. (2024). Scaling Law for Time Series Forecasting. *Advances in Neural Information Processing Systems*, 37.

Spencer R, et al. (2025). Transfer Learning on Transformers for Building Energy Consumption Forecasting — A Comparative Study. *Energy and Buildings*, 336: 115632.

Trindade A (2015). ElectricityLoadDiagrams20112014 [dataset]. UCI Machine Learning Repository.

Vaswani A, et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.

Wilson E, et al. (2022). End-Use Load Profiles for the U.S. Building Stock. Golden, CO: National Renewable Energy Laboratory. Report No.: NREL/TP-5500-80889.

Woo G, et al. (2024). Unified Training of Universal Time Series Forecasting Transformers. *Proceedings of Machine Learning Research*, 235: 53140–53164.

Yao Q, et al. (2025). Towards Neural Scaling Laws for Time Series Foundation Models. In: International Conference on Learning Representations.

Zha D, et al. (2023). Data-centric AI: Perspectives and Challenges. In: Proceedings of the SIAM International Conference on Data Mining, pp. 945–948.

---

*Declarations, acknowledgements, and author information are provided on the separate Title Page as required by the double-blind review process.*

---

## Appendix A: Multi-Seed Results

### A.1 Korean-700 RevIN ON (s = 18,000, 5 seeds)

| Seed | NRMSE (%) | NCRPS (%) |
|:----:|:----------:|:---------:|
| 42 | 12.93 | 7.10 |
| 43 | 13.06 | 7.16 |
| 44 | 13.10 | 7.16 |
| 45 | 13.39 | n.e. |
| 46 | 13.07 | 7.14 |
| **Mean ± Std** | **13.11 ± 0.17** | **7.14 ± 0.03‡** |

‡ Four-seed mean (seeds 42, 43, 44, 46); seed 45 NCRPS not evaluated with the retrained checkpoint used for NRMSE.

### A.2 Korean-700 RevIN OFF (s = 18,000, 3 seeds)

| Seed | NRMSE (%) | NCRPS (%) |
|:----:|:----------:|:---------:|
| 42 | 14.81 | 8.17 |
| 43 | 14.94 | n.e. |
| 44 | 14.40 | 8.40 |
| **Mean ± Std** | **14.72 ± 0.28** | **8.29§** |

§ Two-seed mean (seeds 42, 44); seed 43 NCRPS not evaluated.

### A.3 Korean-700 RevIN ON (s = 16,000, 5 seeds)

Five-seed evaluation at s = 16,000 yields 13.13 ± 0.15%, confirming robustness to training duration.

| Seed | NRMSE (%) |
|:----:|:----------:|
| 42 | 12.89 |
| 43 | 13.24 |
| 44 | 13.16 |
| 45 | 13.26 |
| 46 | 13.12 |
| **Mean ± Std** | **13.13 ± 0.15** |

### A.4 Korean-700 RevIN ON, No Augmentation (s = 18,000, 3 seeds)

| Seed | NRMSE (%) | NCRPS (%) |
|:----:|:----------:|:---------:|
| 42 | 13.48 | 7.95 |
| 43 | 13.89 | 8.21 |
| 44 | 13.65 | 8.31 |
| **Mean ± Std** | **13.67 ± 0.21** | **8.16 ± 0.18** |

### A.5 Stock-Model 900K + RevIN

The 900K stock-model baseline retrained with RevIN achieves 13.89%, a 0.62 pp degradation relative to the original 13.27%.

---

## Appendix B: Stock-Model Scaling (700 and 7,000 Buildings)

| Configuration | N | RevIN | Aug | NRMSE (%) | NCRPS (%) |
|--------------|:---:|:-----:|:--:|:----------:|:---------:|
| Stock-700 | 700 | ON | OFF | 15.28 | — |
| Stock-700 (aug) | 700 | ON | ON | 14.26 | 7.80 |
| Stock-700 | 700 | OFF | OFF | 16.44 | — |
| Stock-7K | 7,000 | ON | OFF | 14.50 | — |
| Stock-7K | 7,000 | OFF | OFF | 15.41 | — |

A 10× increase in stock-model buildings (700 to 7,000) reduces NRMSE by only 0.78 pp, and even 7,000 stock-model buildings (14.50%) do not approach 700 LHS-designed buildings (13.11%).

---

## Appendix C: Additional Ablation Results (Earlier Pipeline)

| Experiment | NRMSE (%) | Δ baseline | Mechanism |
|------------|:----------:|:----------:|-----------|
| BB Box-Cox | 16.24 | +3.31 | Distribution mismatch |
| 4× tokens (168K steps) | 16.02 | +3.09 | Synthetic overfitting |
| 5K cap (70K buildings) | 15.35 | +2.42 | Fixed-budget undersampling |
| Seasonal decomp + RevIN | 16.65 | +3.72 | Periodicity disruption |

---

## Figures

**Fig. 1.** End-to-end pipeline: from building archetype selection and 12D LHS parameter sampling through EnergyPlus simulation, Box-Cox normalization, RevIN-equipped Transformer training, to zero-shot inference without geographic information.

**Fig. 2.** Equal-scale comparison: NRMSE (%) of Korean-700, BB-700, and BB-900K on the 955-building evaluation set. The critical comparison is Korean-700 vs. BB-700 (aug-matched) at identical N = 700, comparing data-design effects under matched training conditions. Dashed line: BB-900K baseline (13.27%).

**Fig. 3.** N-scaling curve showing NRMSE as a function of training buildings (seed = 42). Diminishing returns are evident beyond 70 buildings (n = 5), consistent with pattern-coverage-governed learning.

**Fig. 4.** RevIN's regime-dependent effect across dataset scales. Green arrows: improvement; red: degradation. The asymmetry between small diverse datasets and the full 900K corpus illustrates regime-dependent normalization.
