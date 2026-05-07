# The Operational Diversity Hypothesis: Why Pattern Coverage, Not Corpus Scale, Governs Zero-Shot Building Load Forecasting

Jeong-Uk Kim

Department of Electrical Engineering, Sangmyung University, Seoul 03016, South Korea

ORCID: 0000-0001-9576-8757

E-mail: jukim@smu.ac.kr

---

## Abstract

Foundation models for building energy forecasting are trained on massive synthetic corpora under the implicit assumption that scale drives generalization. We challenge this assumption by proposing the Operational Diversity Hypothesis: zero-shot forecasting accuracy is governed primarily by the coverage of distinct temporal load patterns in the training set, not by the number of buildings. We test this hypothesis through a controlled experiment comparing two training corpora of equal size (700 buildings each): one sampled from the 900,000-building Buildings-900K stock-model corpus, and one constructed via 12-dimensional Latin Hypercube Sampling (LHS) over operational schedule parameters. Both train the same Transformer architecture under identical conditions. The LHS-designed corpus achieves 13.11 ± 0.17% NRMSE (five-seed mean) on 955 real U.S. and Portuguese commercial buildings—matching the full 900,000-building baseline (13.27%)—while the stock-model sample of equal size under identical training conditions reaches only 14.26% (seed 42). An n-scaling analysis reveals that performance saturates at 70 buildings, consistent with a learning regime governed by pattern coverage rather than sample count. RevIN is beneficial in small diverse datasets but degrades the full 900K corpus, demonstrating regime-dependent normalization effects. A U.S.-weather ablation preserves the advantage over equal-scale controls, ruling out climate origin as the sole driver. These findings reframe the data requirement for zero-shot building load forecasting from a scale problem to a design problem, with immediate practical implications: a single workstation day of simulation suffices where a national building stock database was previously assumed necessary.

**Keywords**: operational diversity, zero-shot forecasting, building energy, foundation models, data-centric AI, Latin Hypercube Sampling, pattern coverage

---

---

## 1. Introduction

### 1.1 The Scale Assumption in Building Energy Foundation Models

Short-term load forecasting underpins grid balancing, demand response, and real-time building energy management [1, 2]. Foundation models for time series—trained once and transferred to unseen targets—promise to replace the classical per-building approach. BuildingsBench [3] operationalized this for the building domain by assembling Buildings-900K, a corpus of roughly 900,000 synthetic buildings from the NREL End-Use Load Profiles database [4], and training a Transformer with Gaussian NLL loss. The result—13.28% median NRMSE on 955 real U.S. and Portuguese commercial buildings (which we reproduce at 13.27%)—established both a benchmark and an implicit assumption: that large-scale synthetic corpora are required for zero-shot generalization.

This assumption parallels the scaling paradigm in natural language processing, where more data consistently yields better models. However, recent work has begun to question whether this paradigm transfers to time series. Shi et al. [5] showed that scaling laws for time series forecasting diverge from those in language modeling: more capable models do not always outperform less capable ones, and performance depends on the interaction between look-back horizon, autocorrelation, and non-stationarity. MOIRAI-MoE [6] achieved superior zero-shot performance through architectural specialization rather than increased scale. These findings suggest that the relationship between data volume and generalization may be fundamentally different in time series.

### 1.2 The Operational Diversity Hypothesis

We propose a specific alternative to the scale assumption: the **Operational Diversity Hypothesis**. In zero-shot building load forecasting, prediction accuracy is governed primarily by the diversity of temporal load patterns in the training set—the coverage of the operational schedule space—rather than by the number of training buildings.

The hypothesis rests on a structural observation about stock-model corpora. Buildings-900K, despite its size, draws from a single national stock-model generation pipeline. Buildings in such a corpus share common operational assumptions: most offices operate 08:00–18:00 on weekdays; most hospitals run continuously; most retail stores have moderate baseloads. Adding more buildings from this distribution increases sample size but not necessarily the range of temporal dynamics the model encounters. Many of the 900,000 buildings occupy a narrow region of the space of possible load profiles.

An alternative is to construct a small training set with maximal operational diversity by design. If temporal pattern coverage—not sample count—is the operative factor, then a carefully designed corpus of hundreds of buildings should suffice where hundreds of thousands were previously assumed necessary.

### 1.3 Experimental Approach and Contributions

We test this hypothesis through a controlled experiment. We generate 700 EnergyPlus simulations (50 per archetype × 14 building types) with operational schedules sampled via 12-dimensional Latin Hypercube Sampling. The 12 parameters span operating hours, baseload characteristics, equipment retention, weekly disruptions, seasonal variation, and stochastic noise—designed to cover the space of plausible commercial building operations. Combined with Reversible Instance Normalization (RevIN) [8] and data augmentation, this corpus trains the identical Transformer used in BuildingsBench.

The key innovation is not the use of LHS or RevIN individually—both are established techniques—but the demonstration that their combination reframes the data requirement for zero-shot building energy forecasting from a scale problem to a design problem. Specifically:

1. **The Operational Diversity Hypothesis is empirically supported.** At equal scale (700 buildings each), LHS-designed simulations outperform stock-model samples by 1.33 percentage points under identical training conditions (seed-42: 12.93% vs. 14.26%), isolating data design as the primary factor. The five-seed mean (13.11 ± 0.17%) matches the full 900K-building baseline (13.27%).

2. **Pattern coverage, not sample count, governs accuracy.** N-scaling analysis shows performance saturating at 70 buildings (5 per archetype). This is inconsistent with sample-count-driven learning and consistent with a regime where accuracy depends on the number of distinct temporal patterns.

3. **RevIN exhibits a regime-dependent normalization effect.** RevIN improves small diverse datasets (−1.6 pp) but degrades the full 900K corpus (+0.62 pp). This asymmetry has a mechanistic explanation rooted in the information content of load magnitude across data regimes, with implications beyond building energy.

4. **Cross-climate transfer rules out geographic confounds.** Korean-weather simulations achieve benchmark-level accuracy on U.S. and Portuguese loads. A U.S.-weather ablation using the same LHS schedules preserves the advantage over equal-scale controls, isolating schedule diversity from climate origin.

---

## 2. Related Work

### 2.1 Building Energy Forecasting

Prior methods ranging from statistical models [9] to gradient boosting [10], recurrent networks [11], and Transformers [12, 13] improved per-building accuracy but remained building-specific. Transfer learning through domain adaptation [14] and fine-tuning [15] relaxes this constraint but still requires target-domain data. Our work eliminates the need for target-building data through zero-shot inference, while demonstrating that the training corpus can be radically smaller than previously assumed.

### 2.2 BuildingsBench and the Scale Paradigm

BuildingsBench [3] framed building load forecasting as a zero-shot generalization problem. The training corpus, Buildings-900K, comprises approximately 350,000 commercial (ComStock) and 550,000 residential (ResStock) synthetic buildings generated by the NREL EULP pipeline [4]. The model trains for approximately 0.067 epochs—roughly 2–3 sampled training windows per building—using fitted global Box-Cox normalization (λ = −0.067). The sub-epoch training regime is consequential: the authors found that additional training degraded OOD performance, interpreted as evidence that the model memorizes synthetic patterns faster than it learns transferable representations.

Our work keeps the model architecture and evaluation protocol fixed while changing the training data and normalization strategy. The comparison tests whether the same architecture can achieve comparable performance with a fundamentally different—and radically smaller—training corpus.

### 2.3 Scaling Laws and Data Efficiency in Time Series

The proliferation of general-purpose time series foundation models—TimesFM [16], Chronos [17], Lag-Llama [18], MOIRAI [19]—has established that large-scale pretraining enables zero-shot forecasting across domains. Yao et al. [20] found that model architecture significantly influences scaling efficiency. Most relevant to our work, Shi et al. [5] showed that in time series, additional data does not expand the combinatorial coverage of contexts as directly as additional tokens do in language modeling. Our findings provide a concrete, domain-specific instance of this principle: in building energy, the marginal information content of additional stock-model buildings decreases rapidly because they occupy similar regions of operational schedule space.

### 2.4 Data-Centric AI and Instance Normalization

The data-centric AI perspective [7] holds that improving data quality is often more productive than improving model architecture. Our 12-dimensional LHS design applies this principle to the energy domain: rather than sampling from a fixed distribution, we construct a training set that fills the space of plausible operations.

RevIN [8] normalizes instance-level load magnitude and variability before the encoder and restores the scale after prediction. Our finding that RevIN's benefit is regime-dependent—helpful for small diverse datasets, harmful at large scale—extends the understanding of when instance normalization helps versus hurts, connecting to broader questions about the role of magnitude information in time series models.

---

## 3. Method

### 3.1 Parametric Building Simulation with LHS-Designed Operational Diversity

Training data are generated through EnergyPlus [21] simulation of commercial buildings with parametrically varied operational schedules. The goal is not to replicate a building stock distribution but to maximize the coverage of the temporal pattern space.

**Building archetypes.** We use 14 building archetypes based on DOE commercial reference building models [22] adapted to Korean building codes and climate zones: office, retail, school, hotel, hospital, apartment (midrise and highrise), small office, large office, warehouse, strip mall, restaurant (full-service and quick-service), and university. Each archetype begins from a DOE reference model with code-compliant envelope thermal properties, HVAC system sizing, and internal load densities.

**Climate coverage.** Simulations span five Korean cities across three climate zones: Seoul (central), Busan (southern), Daegu (southern), Gangneung (central-coastal), and Jeju (subtropical). For the U.S.-TMY ablation (Section 4.2), the same 700 schedule samples and archetypes were rerun with U.S. TMY files mapped by approximate ASHRAE climate-zone similarity: Seoul → Washington DC (4A), Busan → Atlanta (3A), Daegu → Charlotte (3A), Gangneung → Boston (5A), and Jeju → Miami (1A).

**Operational parameter space.** Building operational schedules are parameterized by 12 continuous variables sampled via Latin Hypercube Sampling [23] (Table 1). Unlike stock-model sampling—which reflects the statistical distribution of real buildings (many near typical conditions, few at extremes)—LHS ensures uniform marginal coverage across all dimensions. Stock-model sampling asks "what does the building stock look like?"; LHS asks "what temporal patterns can buildings produce?"

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

The resulting training set ranges from 24/7 high-baseload facilities (resembling data centers or hospitals) to weekday-only offices with steep morning ramps and low overnight loads. For the n = 50 configuration used in our main experiments, this yields 50 × 14 = 700 buildings. Each sample modifies the EnergyPlus IDF file, runs a full annual simulation (8,760 hours), and extracts hourly total electricity consumption.

**Post-hoc distributional validation.** Table 2 provides a post-hoc comparison showing that including the four diversity parameters (night_equipment_frac, weekly_break_prob, seasonal_amplitude, process_load_frac) brings simulated profiles closer to real-building statistics. These statistics were computed after defining the parameter space and were not used for model selection or hyperparameter tuning.

**Table 2.** Post-hoc comparison of simulated and real-building load-profile statistics. Metrics characterize input sequence shape (X), not prediction targets (y).

| Metric | Without 4 params | With 4 params | BB Real Buildings |
|--------|:----------------:|:-------------:|:-----------------:|
| Night/Day ratio | 0.512 | 0.852 | 0.803 |
| Autocorrelation (168h) | 0.938 | 0.784 | 0.751 |
| Baseload P95 | 0.51 | 0.905 | 0.725 |
| CV P5 (flatness) | 0.21 | 0.032 | 0.105 |

### 3.2 Model Architecture

To isolate the effect of training data from model design, we adopt the identical Transformer architecture used in BuildingsBench [3]: 3 encoder and 3 decoder layers, d_model = 512, 8 attention heads, feedforward dimension 1024, totaling 15.8M parameters (Transformer-M). Input consists of 168 hourly load values with temporal features (day-of-year sinusoidal encoding, day-of-week and hour-of-day embeddings). The model predicts 24-hour-ahead Gaussian distributions through autoregressive decoding with Gaussian NLL loss.

**Reversible Instance Normalization.** Before encoding, the 168-hour context window is normalized by its instance mean and standard deviation; after decoding, the normalization is reversed. This removes building-specific load magnitude and variability, allowing the model to focus on temporal shape. We apply RevIN symmetrically in all controlled experiments, including BuildingsBench baselines, to ensure fair comparison. The comparison between our RevIN-equipped model and the original BuildingsBench (which does not use RevIN) is discussed in Section 5.1.

**No geographic features.** BuildingsBench provides latitude and longitude embeddings as model inputs. We set both to zero for all buildings—both training and evaluation. The ablation in Section 4.4 shows that actual Korean coordinates provide no benefit when evaluation coordinates are unavailable, consistent with RevIN absorbing the scale differences that coordinates might otherwise encode.

### 3.3 Training Protocol

We apply a global Box-Cox transform [24] fitted on our simulation data (λ = −0.067). AdamW [25] with learning rate 6 × 10⁻⁵, weight decay 0.01, cosine annealing, and 500-step warmup. Training runs for 18,000 gradient steps with batch size 128, corresponding to approximately 9 epochs over 700 buildings. Data augmentation includes window jitter (±1–6h), Gaussian noise (σ = 0.02 in Box-Cox space), and amplitude scaling (U[0.85, 1.15]).

Both our training and BuildingsBench use approximately 18,000 gradient steps, but distributed differently: each of our 700 buildings is seen roughly 3,300 times (across window positions), while each BuildingsBench building contributes roughly 2–3 windows. Three factors mitigate overfitting: (1) augmentation ensures each exposure presents a different view; (2) RevIN prevents memorizing absolute load levels; (3) the high inter-building diversity of LHS ensures patterns learned from any one building generalize.

### 3.4 Evaluation Protocol

We follow the BuildingsBench test set, sliding-window construction, and aggregation metric without modification. The test set comprises 955 real load series: 611 from U.S. commercial buildings in the Building Data Genome Project 2 (BDG-2) [26], drawn from four university and government campuses spanning Mediterranean, semi-arid, humid subtropical, and humid continental climates, and 344 from Portuguese electricity consumers [27], with 15 out-of-vocabulary buildings excluded per the original specification. Per-building NRMSE = sqrt(MSE) / mean(actual), aggregated by median across buildings. The sliding window uses stride 24 hours, context 168 hours, and prediction horizon 24 hours. We reproduced the BuildingsBench baseline using the official checkpoint and obtained 13.27%, confirming pipeline equivalence within 0.01 pp of the reported 13.28%.

---

## 4. Experiments and Results

### 4.1 Equal-Scale Comparison: Isolating the Data-Design Effect

The central experiment compares two 700-building corpora under identical training conditions, isolating data design as the independent variable (Table 3, Fig. 2).

**Table 3.** Main results on the 955-building evaluation set. Korean-700 and US-TMY-700 report five-seed mean ± std; Korean-700 (no aug) and Korean-700 (RevIN OFF) report three-seed mean ± std; all other rows use seed = 42. †BB-900K NCRPS unavailable due to Box-Cox protocol mismatch. BB-700 (aug-matched) uses identical optimizer, augmentation, and steps as Korean-700.

| Model | Data | N | RevIN | Aug | NRMSE (%) | NCRPS (%) | Δ baseline |
|-------|------|------:|:-----:|:---:|:---------:|:---------:|:------:|
| BB-900K | BB 900K | 900,000 | OFF | OFF | 13.27 | —† | — |
| BB+RevIN | BB 900K | 900,000 | ON | OFF | 13.89 | 7.76 | +0.62 |
| **K-700 (ours)** | **LHS sim** | **700** | **ON** | **ON** | **13.11 ± 0.17** | **7.14 ± 0.03** | **−0.16** |
| K-700 no aug | LHS sim | 700 | ON | OFF | 13.67 ± 0.21 | 8.16 | +0.40 |
| K-700 no RevIN | LHS sim | 700 | OFF | ON | 14.72 ± 0.28 | 8.29 | +1.45 |
| BB-700 aug | BB subset | 700 | ON | ON | 14.26 | 7.80 | +0.99 |
| **US-700 (ours)** | **US TMY sim** | **700** | **ON** | **ON** | **13.64 ± 0.65** | **7.53 ± 0.41** | **+0.37** |
| BB-700 | BB subset | 700 | ON | OFF | 15.28 | — | +2.01 |
| BB-700 OFF | BB subset | 700 | OFF | OFF | 16.44 | — | +3.17 |
| Persist. | — | — | — | — | 16.68 | — | +3.41 |

The equal-scale comparison is the critical test of the Operational Diversity Hypothesis. Under identical conditions (same architecture, same optimizer, same augmentation, same number of buildings, same RevIN), the LHS-designed corpus outperforms the stock-model sample by **1.33 pp** at seed 42 (12.93% vs. 14.26%). Since BB-700 aug is a single-seed result, the fairest comparison uses seed-matched values; across all conditions, the data-design gap consistently exceeds 1 pp. This gap cannot be attributed to RevIN, augmentation, model architecture, or training budget—all are controlled. The remaining variable is the design of the training data.

Korean-700 (five-seed mean 13.11 ± 0.17%) numerically matches the full BB-900K baseline (13.27%), achieving comparable accuracy with 1,286× fewer buildings and no geographic metadata. A paired per-building comparison (Korean-700 five-seed average vs. BB-900K) shows Korean-700 achieves lower error on 680 of 955 buildings (71%); paired bootstrap 95% CI of the median difference is [0.31, 0.39] pp in favor of Korean-700 (Wilcoxon signed-rank p < 0.001).

### 4.2 Climate Ablation: Isolating Schedule Diversity from Weather

To isolate schedule diversity from climate effects, we retrained on the same 700 LHS-designed buildings using U.S. TMY weather (Seoul → Washington DC, Busan → Atlanta, Daegu → Charlotte, Gangneung → Boston, Jeju → Miami). The five-seed mean of US-TMY-700 is 13.64 ± 0.65%, within 0.53 pp of Korean-700. The U.S.-TMY model still outperforms BB-700 (14.26%) by 0.62 pp, demonstrating that LHS-designed schedule diversity contributes substantially beyond climate origin. The higher seed variance (0.65% vs. 0.17%) indicates that weather choice affects run-to-run stability.

### 4.3 N-Scaling: Evidence for Pattern-Count-Governed Learning

If the Operational Diversity Hypothesis is correct, performance should saturate once the LHS parameter space is adequately covered, regardless of further increases in sample count. Table 4 and Fig. 3 test this prediction.

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

Performance improves sharply from n = 1 to n = 5 (14.72% to 13.28%), matching the BB-900K baseline (13.27%) at just 70 buildings. Beyond n = 5, gains become incremental—fluctuations within ±0.15 pp rather than systematic improvement. This saturation profile is qualitatively different from the monotonic improvement predicted by power-law scaling and is consistent with a learning regime governed by the number of distinct temporal patterns rather than the number of examples.

The contrast with BuildingsBench scaling is instructive (Appendix B): increasing from 700 to 7,000 stock-model buildings improves NRMSE by only 0.78 pp (15.28% to 14.50%), and even BB-7K does not approach 700 LHS-designed buildings (13.11%). Adding more stock-model buildings provides diminishing returns because they occupy similar regions of operational space.

### 4.4 Ablation Studies

**Table 5.** Ablation results (val_best checkpoint, 955 buildings).

| Experiment | NRMSE (%) | Delta | Interpretation |
|------------|:----------:|:-----:|----------------|
| K-700 RevIN ON (baseline) | 12.93 | — | Best seed (seed 42, lat/lon = zero) |
| RevIN OFF (3-seed mean) | 14.72 | +1.79 | RevIN contributes 1.79 pp |
| Actual lat/lon coordinates | 12.93 | 0.00 | Geographic metadata provides no benefit |

Additional ablations (Appendix C) show that using BB-fitted Box-Cox parameters (+3.31 pp), extended training (+3.09 pp), scaling to 70K buildings with fixed budget (+2.42 pp), and seasonal decomposition before RevIN (+3.72 pp) all degrade performance, each illustrating a distinct failure mode.

---

## 5. Discussion

### 5.1 Mechanistic Explanation: RevIN's Regime-Dependent Effect

The asymmetry in RevIN's effect across dataset scales (Fig. 4) has a clear mechanistic explanation that extends beyond our specific setting.

RevIN normalizes each context window to zero mean and unit variance, stripping absolute load magnitude. With 700 LHS-designed buildings, the model has not seen enough magnitude variation to internalize it; RevIN solves this analytically, yielding a 1.6 pp improvement. With 900,000 stock-model buildings—each contributing roughly 2–3 training windows—the model has already learned to exploit magnitude as a signal: a building consuming 500 kW at midnight behaves differently from one consuming 5 kW. Removing this information discards something useful, producing a 0.62 pp degradation.

A structural factor reinforces this. In Buildings-900K, magnitude is informative about building type because HVAC, envelope, and schedule are jointly determined by stock-model parameters—magnitude and temporal shape are correlated. In our LHS design, magnitude reflects only the scale_mult parameter, sampled independently of schedule parameters. RevIN's removal of magnitude therefore discards less useful information in LHS data than in stock-model data.

This finding has implications for the broader time series community: RevIN's benefit depends not just on dataset size but on the information structure of the training data—specifically, whether magnitude carries predictive information about temporal dynamics.

### 5.2 Why Operational Diversity Governs Accuracy

The Operational Diversity Hypothesis provides a parsimonious explanation for four otherwise disconnected observations:

1. **Equal-scale superiority.** LHS-700 outperforms BB-700 because LHS achieves broader pattern coverage with fewer buildings. The 12-dimensional design produces buildings spanning the full range of plausible operations, while 700 stock-model buildings cluster around typical conditions.

2. **Rapid n-scaling saturation.** Performance plateaus at 70 buildings because the LHS parameter space is adequately covered with 5 samples per archetype. Additional samples refine existing patterns rather than introducing new ones.

3. **Sub-epoch optimality in BuildingsBench.** The finding that BuildingsBench achieves best OOD performance with less than one epoch is consistent with our hypothesis: the useful information in 900K buildings—the distinct temporal patterns—is learned quickly, and further training memorizes stock-model-specific artifacts.

4. **Climate robustness.** The U.S.-TMY ablation preserves the data-design advantage because operational schedule diversity, not climate-specific features, drives the generalization. Weather affects thermal loads, but the temporal dynamics—operating hours, ramp patterns, baseload levels—are schedule-driven.

From a scaling-law perspective, this is consistent with Shi et al. [5]: in building load time series, each additional building from the same stock model may add only a variation on patterns already in the training set. The marginal information content decreases rapidly when the operational diversity of the corpus is fixed.

### 5.3 Practical Implications

The reframing from scale to design has immediate practical consequences. An organization can generate several hundred parametric EnergyPlus simulations conforming to local building codes and climate, train a standard Transformer with RevIN and augmentation, and deploy zero-shot forecasting—without assembling a national building stock database.

The computational cost is modest: 700 simulations complete in approximately 4 hours on a single workstation; training takes approximately 2 hours on a single GPU. The entire pipeline from simulation to deployed model takes less than a day. No location metadata, building-type classification, or HVAC information is required at inference time—only 168 hours of historical load data.

### 5.4 Limitations

Our parametric simulations differ from BuildingsBench in schedule design, climate, building codes, and envelope parameters. The equal-scale controlled experiment (Section 4.1) isolates the data-source effect under matched augmentation, but cannot fully disentangle all confounds. The U.S.-TMY ablation substantially reduces but does not eliminate the climate-origin confound.

The augmentation applied to Korean-700 (window jitter, Gaussian noise, amplitude scaling) was not part of the original BuildingsBench training pipeline. The aug-matched BB-700 control addresses this asymmetry at equal scale, but the comparison between Korean-700 and the original BB-900K (which uses no augmentation) includes this confound. Augmentation contributes 0.56 pp to Korean-700 (13.67% → 13.11%); even without augmentation, Korean-700 (13.67%) outperforms BB-700 aug (14.26%), but falls above the BB-900K baseline (13.27%).

The aggregate improvement over the BB-900K baseline (0.16 pp) is modest and comparable to inter-seed variation (0.17%). The paired per-building comparison confirms statistical significance (p < 0.001, 71% win rate), but the margin is small. The claim is not that Korean-700 dramatically exceeds the baseline, but that it matches the baseline with 1,286× fewer buildings—supporting the hypothesis that pattern coverage, not scale, is the operative factor.

All experiments use Transformer-M (15.8M parameters); generalization to PatchTST, MOIRAI, or other architectures has not been tested. On residential buildings (953 buildings), our model yields 77.71% NRMSE, comparable to the Persistence Ensemble (77.88%), confirming that zero-shot residential forecasting remains an open problem. Real-world validation beyond the BuildingsBench evaluation set has not been conducted; broader testing across diverse building types in multiple countries is needed.

---

## 6. Conclusion

We proposed and tested the Operational Diversity Hypothesis: zero-shot building load forecasting accuracy is governed by the coverage of temporal load patterns in the training set, not by corpus scale. Four lines of evidence support this hypothesis: (1) equal-scale controlled experiments show a persistent data-design advantage exceeding 1 pp; (2) n-scaling saturates at 70 buildings; (3) RevIN's regime-dependent effect is mechanistically consistent with pattern-coverage-governed learning; and (4) cross-climate transfer confirms that schedule diversity, not climate origin, drives generalization.

The practical implication is a reframing of the data requirement: from assembling a national building stock database to designing a few hundred simulations with maximal operational diversity. This reduces the barrier to deploying zero-shot building energy forecasting in data-sparse regions from a data-engineering problem requiring access to national building stock models to a simulation design problem solvable on a single workstation.

The principal open questions are the limitation to commercial buildings, the augmentation asymmetry between pipelines, and the untested generalization to other model architectures. Extending the operational diversity framework to residential buildings—where zero-shot performance remains at Persistence-level—is a natural next step.

---

## Acknowledgements

This work was supported by the Korea Institute of Energy Technology Evaluation and Planning (KETEP) grant funded by the Ministry of Trade, Industry and Energy, Republic of Korea (Grant No. RS-00238487).

## CRediT Author Statement

**Jeong-Uk Kim**: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review & editing, Visualization.

## Data Availability

The simulation pipeline, model source code, and training configurations are publicly available at https://github.com/jukkim/korean-buildingsbench. The 700-building parametric simulation dataset can be regenerated from the provided pipeline or is available from the corresponding author upon request. Pretrained checkpoints for the primary results (Korean-700 seeds 42–46, BB-700, BB 900K + RevIN, US-TMY-700 seeds 42–46) and n-scaling intermediate points are available from the corresponding author upon request. The BuildingsBench evaluation data can be downloaded from the NREL Open Energy Data Initiative (https://data.openei.org/submissions/5859) under a CC-BY 4.0 license.

## Declaration of Competing Interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation Process

During the preparation of this work the author used Claude (Anthropic) in order to assist with code development for simulation post-processing, figure generation scripts, and manuscript formatting. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

---

## References

[1] Fan C, et al. A short-term building cooling load prediction method using deep learning algorithms. Appl Energy 2017;195:222–33.

[2] Hong T, et al. Ten questions on urban building energy modeling. Build Environ 2019;168:106508.

[3] Emami P, Sahu A, Graf P. BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting. Adv Neural Inf Process Syst 2023;36.

[4] Wilson E, et al. End-Use Load Profiles for the U.S. Building Stock. Golden (CO): National Renewable Energy Laboratory; 2022. Report No.: NREL/TP-5500-80889.

[5] Shi J, et al. Scaling Law for Time Series Forecasting. Adv Neural Inf Process Syst 2024;37.

[6] Liu X, et al. MOIRAI-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts. Proc Mach Learn Res 2025;267:38940–62.

[7] Zha D, et al. Data-centric AI: Perspectives and Challenges. Proc SIAM Int Conf Data Min; 2023. p. 945–8.

[8] Kim T, et al. Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. Int Conf Learn Represent; 2022.

[9] Amasyali K, El-Gohary NM. A review of data-driven building energy consumption prediction studies. Renew Sustain Energy Rev 2018;81:1192–205.

[10] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. Proc 22nd ACM SIGKDD Int Conf Knowl Discov Data Min; 2016. p. 785–94.

[11] Hochreiter S, Schmidhuber J. Long Short-Term Memory. Neural Comput 1997;9(8):1735–80.

[12] Vaswani A, et al. Attention Is All You Need. Adv Neural Inf Process Syst 2017;30.

[13] Nie Y, et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. Int Conf Learn Represent; 2023.

[14] Ribeiro M, et al. Transfer learning with seasonal and trend adjustment for cross-building energy forecasting. Energy Build 2018;165:352–63.

[15] Spencer R, et al. Transfer Learning on Transformers for Building Energy Consumption Forecasting — A Comparative Study. Energy Build 2025;336:115632.

[16] Das A, et al. A decoder-only foundation model for time-series forecasting. Proc Mach Learn Res 2024;235:10148–67.

[17] Ansari AF, et al. Chronos: Learning the Language of Time Series. Trans Mach Learn Res 2024.

[18] Rasul K, et al. Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting. NeurIPS Workshop Robustness Found Models; 2023.

[19] Woo G, et al. Unified Training of Universal Time Series Forecasting Transformers. Proc Mach Learn Res 2024;235:53140–64.

[20] Yao Q, et al. Towards Neural Scaling Laws for Time Series Foundation Models. Int Conf Learn Represent; 2025.

[21] Crawley DB, et al. EnergyPlus: creating a new-generation building energy simulation program. Energy Build 2001;33(4):319–31.

[22] Deru M, et al. U.S. Department of Energy Commercial Reference Building Models of the National Building Stock. Golden (CO): National Renewable Energy Laboratory; 2011. Report No.: NREL/TP-5500-46861.

[23] McKay MD, Beckman RJ, Conover WJ. A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. Technometrics 1979;21(2):239–45.

[24] Box GEP, Cox DR. An Analysis of Transformations. J R Stat Soc B 1964;26(2):211–52.

[25] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. Int Conf Learn Represent; 2019.

[26] Miller C, et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. Sci Data 2020;7:368.

[27] Trindade A. ElectricityLoadDiagrams20112014 [dataset]. UCI Machine Learning Repository; 2015.

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

### A.5 BB 900K + RevIN

The BuildingsBench 900K model retrained with RevIN achieves 13.89%, a 0.62 pp degradation relative to the original 13.27%.

---

## Appendix B: BB-700 and BB-7K Scaling

| Configuration | N | RevIN | Aug | NRMSE (%) | NCRPS (%) |
|--------------|:---:|:-----:|:--:|:----------:|:---------:|
| BB-700 | 700 | ON | OFF | 15.28 | — |
| BB-700 (aug) | 700 | ON | ON | 14.26 | 7.80 |
| BB-700 | 700 | OFF | OFF | 16.44 | — |
| BB-7K | 7,000 | ON | OFF | 14.50 | — |
| BB-7K | 7,000 | OFF | OFF | 15.41 | — |

A 10× increase in stock-model buildings (700 to 7,000) reduces NRMSE by only 0.78 pp, and even BB-7K (14.50%) does not approach 700 LHS-designed buildings (13.11%).

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

**Fig. 2.** Equal-scale comparison: NRMSE (%) of Korean-700, BB-700, and BB-900K on the 955-building evaluation set. The critical comparison is Korean-700 vs. BB-700 (aug-matched) at identical N = 700, isolating data design as the independent variable. Dashed line: BB-900K baseline (13.27%).

**Fig. 3.** N-scaling curve showing NRMSE as a function of training buildings (seed = 42). Performance saturates at 70 buildings (n = 5), consistent with pattern-coverage-governed learning.

**Fig. 4.** RevIN's regime-dependent effect across dataset scales. Green arrows: improvement; red: degradation. The asymmetry between small diverse datasets and the full 900K corpus illustrates regime-dependent normalization.
