# Operational Diversity by Design: A Parametric Simulation Methodology for Zero-Shot Building Load Forecasting

Jeong-Uk Kim

Department of Electrical Engineering, Sangmyung University, Seoul 03016, South Korea

ORCID: 0000-0001-9576-8757

E-mail: jukim@smu.ac.kr

---

## Abstract

Zero-shot building load forecasting requires simulation datasets that capture diverse real-world operational patterns. Existing approaches rely on large building-stock databases containing hundreds of thousands of buildings, but the relationship between dataset design and zero-shot generalization remains poorly understood. This paper develops a coverage-based interpretation of the Operational Diversity Hypothesis: zero-shot transfer appears to depend not only on sample count, but also on how broadly the training set spans the operational parameter space.

We define a 12-dimensional operational schedule space — spanning operating hours, baseload behavior, equipment retention, weekly disruptions, seasonal variation, and stochastic noise — and use it to motivate four empirical expectations about the advantage of design-based sampling over stock-model sampling, the onset of N-scaling plateaus, the regime-dependent effect of Reversible Instance Normalization (RevIN), and the failure mode on out-of-scope building types.

We instantiate this interpretation through 700 EnergyPlus simulations (50 per archetype × 14 types) with Latin Hypercube Sampling, combined with RevIN and data augmentation to train a Transformer-based forecasting model. Controlled experiments are broadly consistent with the four expectations: in a controlled single-subset comparison, the designed dataset outperforms an equal-size stock-model sample by 1.33 percentage points; N-scaling plateaus near n ≈ 5 per archetype; RevIN improves LHS-designed data but degrades stock-model data; and residential forecasting remains weak when training covers only commercial operating regimes. These results support a data-centric view of zero-shot building energy forecasting in which simulation design may matter as much as corpus size.

**Keywords**: operational diversity, zero-shot forecasting, building energy, simulation design, data-centric AI, Latin Hypercube Sampling

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

1. **A coverage-based conceptual framework for the Operational Diversity Hypothesis.** We define the operational parameter space, introduce coverage as a useful way to describe zero-shot transferability, and use this lens to derive four empirical expectations about LHS-vs-stock-model performance, N-scaling plateau onset, RevIN's regime-dependent effect, and residential failure modes. The framework is intended as an interpretive model for the experiments rather than as a complete formal proof of generalization.

2. **A data-centric validation framework for simulation design.** We evaluate the designed simulation dataset under matched training conditions, weather-origin ablations, and dataset-size scaling experiments. These controlled comparisons examine whether operational schedule coverage improves zero-shot transferability independently of confounding factors. We use the BuildingsBench evaluation protocol (Emami et al. 2023) as an external reference for zero-shot transfer performance, but the focus of this work is the design of the simulation dataset rather than benchmark competition.

3. **A normalization analysis for designed simulation datasets.** RevIN interacts with the information structure of the training data: it helps when load magnitude is decoupled from schedule dynamics and degrades performance when magnitude carries stock-model-specific information. This regime-dependent effect has implications for the broader time series community.

4. **Practical guidance for compact simulation dataset construction.** The proposed workflow demonstrates that targeted operational diversity can achieve benchmark-level zero-shot transfer, reducing the need for national-scale stock-model pipelines. Weather-origin ablations confirm that the benefit persists across different climate files.

---

## 2. Conceptual Framework

This section introduces a coverage-based conceptual framework for the Operational Diversity Hypothesis. Its purpose is to organize the empirical findings and make the main assumptions explicit, not to claim a complete proof of zero-shot generalization.

### 2.1 Definitions

**Operational parameter space.** The temporal dynamics of a building's load profile are governed by an operational parameter vector $\theta \in \Theta$, where:

$$\Theta = \prod_{i=1}^{d} [\theta_i^{\min}, \theta_i^{\max}] \subset \mathbb{R}^d$$

In this work, $d = 12$ (Table 1). Each $\theta_i$ encodes a schedule dimension such as operating hours, baseload fraction, or equipment retention.

**Load profile mapping.** For building archetype $a \in \mathcal{A}$ (with $|\mathcal{A}| = 14$) and parameter vector $\theta$, EnergyPlus defines a deterministic mapping $g: \mathcal{A} \times \Theta \to \mathcal{X}^{8760}$ producing an annual hourly load series. The forecasting model trains on sliding windows $(x_{t-L+1:t},\; x_{t+1:t+H})$ extracted from these series.

**Temporal pattern.** Applying RevIN to a context window yields a normalized representation $\tilde{x}_t = (x_t - \mu_w) / \sigma_w$, where $\mu_w$ and $\sigma_w$ are the window mean and standard deviation. The functional form of $\tilde{\mathbf{x}}$ defines the temporal pattern $\phi(a, \theta) \in \mathcal{P}$. Because RevIN removes absolute scale, $\phi$ is invariant to the scale_mult parameter:

$$\phi(a, \theta) = \phi(a, \theta') \quad \text{whenever} \quad \theta_i = \theta'_i \;\;\forall\; i \neq i_{\text{scale}}$$

**ε-coverage.** A simulation set $S = \{(a_j, \theta_j)\}_{j=1}^{N}$ achieves ε-coverage of $\Theta$ at level:

$$\mathrm{Cov}_\varepsilon(S, \Theta) = \frac{\lambda\!\left(\bigcup_{j=1}^{N} B_\varepsilon(\theta_j) \cap \Theta\right)}{\lambda(\Theta)}$$

where $B_\varepsilon(\theta) = \{\theta' : \|\theta' - \theta\|_\infty \leq \varepsilon\}$ and $\lambda$ is Lebesgue measure. The minimum covering radius is $\varepsilon^*(S) = \inf\{\varepsilon > 0 : \mathrm{Cov}_\varepsilon(S,\Theta) = 1\}$.

**Pattern diversity.** The diversity of $S$ is the archetype-averaged coverage:

$$\mathcal{D}(S) = \frac{1}{|\mathcal{A}|}\sum_{a \in \mathcal{A}} \mathrm{Cov}_\varepsilon(S_a, \Theta), \quad S_a = \{\theta_j : a_j = a\}$$

### 2.2 Assumptions

**A1 (Lipschitz pattern mapping).** For each archetype $a$, the pattern mapping is Lipschitz continuous:

$$d_{\mathcal{P}}(\phi(a, \theta),\; \phi(a, \theta')) \leq L_a \|\theta - \theta'\|_\infty$$

with global constant $L = \max_a L_a$. This is an interpretive approximation over regions where operational perturbations do not trigger discrete control-regime changes (e.g., HVAC equipment switching or setpoint threshold crossings). Within such regions, EnergyPlus is a physics-based solver: small parameter perturbations produce smooth changes in load profiles within an archetype.

**A2 (Target support).** The target distribution of real-building operating conditions satisfies $\mathrm{supp}(P_{\text{target}}) \subseteq \mathcal{A}_{\text{target}} \times \Theta$, where $\mathcal{A}_{\text{target}} \supseteq \mathcal{A}_{\text{known}}$ may include archetypes not in the training set.

**A3 (Scale-shape independence under RevIN).** After RevIN normalization, the prediction error depends on $\phi(a, \theta)$ but not on the scale_mult value $s$:

$$\mathbb{E}[\ell(\hat{y}, y) \mid \phi(a,\theta)] = \mathbb{E}[\ell(\hat{y}, y) \mid \phi(a,\theta), s]$$

### 2.3 Coverage-Dependent Error Decomposition

**Heuristic relation (informal).** Under A1–A3, the expected zero-shot transfer error of a model $f_S$ trained on simulation set $S$ can be decomposed conceptually as increasing with the covering radius of the training set:

$$\mathcal{E}(f_S) \leq \hat{\mathcal{E}}_S(f_S) + L \cdot \varepsilon^*(S) + \delta_{\mathrm{arch}}$$

where $\hat{\mathcal{E}}_S$ is the empirical training error, $\varepsilon^*(S)$ is the minimum covering radius of $S$, and $\delta_{\mathrm{arch}}$ accounts for target archetypes absent from $\mathcal{A}$.

*Justification.* For a target building $(a^*, \theta^*)$ with $a^* \in \mathcal{A}$, the covering condition guarantees a training sample $\theta_j \in S_{a^*}$ with $\|\theta^* - \theta_j\|_\infty \leq \varepsilon^*(S)$. By the triangle inequality and A1:

$$d_{\mathcal{P}}(\hat\phi_{f_S}, \phi(a^*,\theta^*)) \leq \underbrace{d_{\mathcal{P}}(\hat\phi_{f_S}, \phi(a^*,\theta_j))}_{\leq\;\hat{\mathcal{E}}_S} + \underbrace{d_{\mathcal{P}}(\phi(a^*,\theta_j), \phi(a^*,\theta^*))}_{\leq\;L\,\varepsilon^*(S)}$$

For $a^* \notin \mathcal{A}$, the additional error is captured by $\delta_{\mathrm{arch}}$, the maximum pattern distance to the nearest training archetype. This decomposition is intended as an interpretive organizing principle rather than a rigorous generalization bound; the relationship between pattern-space distance $d_{\mathcal{P}}$ and forecasting loss $\ell$ is assumed rather than formally derived.

The key feature of this expression is that the coverage term $L \cdot \varepsilon^*(S)$ depends on how well $S$ fills the parameter space, not only on the number of samples $N = |S|$. Two datasets with the same $N$ but different coverage can therefore be expected to generalize differently.

### 2.4 Theoretical Predictions

The framework suggests four testable expectations:

**Prediction 1: LHS outperforms stock-model sampling at equal scale.** If two datasets have the same size but one spreads samples more broadly across the operational parameter space, the broader set should transfer better. This expectation is tested in Section 5.1 through K-700 versus BB-700.

**Prediction 2: N-scaling plateau.** In a 12-dimensional schedule space, the covering radius decreases heuristically as $n^{-1/12}$ per archetype — an extremely slow function (doubling $n$ reduces $\varepsilon^*$ by only ~6%). Increasing per-archetype sample count should therefore produce rapidly diminishing returns once coarse coverage is achieved. This expectation is tested in Section 5.3.

**Prediction 3: RevIN helps LHS, hurts stock-model.** When magnitude is sampled largely independently from temporal shape, RevIN should help by suppressing nuisance scale variation. When magnitude is entangled with other informative features, RevIN may remove useful signal. This expectation is tested in Section 5.5 and Table 3.

**Prediction 4: Residential failure as coverage gap.** If residential operating regimes lie largely outside the commercial parameter space used for training, poor transfer should be expected. This is discussed in Section 6.4.

---

## 3. Related Work
### 3.1 Building Energy Forecasting and Zero-Shot Transfer

Building energy forecasting has progressed from statistical models (Amasyali and El-Gohary 2018) through gradient boosting (Chen and Guestrin 2016), recurrent networks (Hochreiter and Schmidhuber 1997), and Transformers (Vaswani et al. 2017; Nie et al. 2023). These methods achieve high per-building accuracy but require target-building training data. Transfer learning through domain adaptation (Ribeiro et al. 2018) and fine-tuning (Spencer et al. 2025) relaxes this constraint but still needs target-domain samples.

The zero-shot paradigm eliminates target-building data entirely: a model trained on synthetic simulations generalizes to unseen real buildings at inference time. Emami et al. (2023) operationalized this through a 900K-building corpus drawn from the NREL EULP pipeline (Wilson et al. 2022), training a Transformer with Gaussian NLL loss for approximately 0.067 epochs—roughly 2–3 sampled windows per building—using global Box-Cox normalization (λ = −0.067). The sub-epoch training regime is consequential: additional training degraded out-of-distribution performance, which the authors interpreted as evidence that the model memorizes synthetic patterns faster than it learns transferable representations. This observation is consistent with the Operational Diversity Hypothesis—the useful information may be learned quickly because many buildings produce similar temporal patterns.

### 3.2 Scaling Laws and Data Efficiency in Time Series

The proliferation of general-purpose time series foundation models—TimesFM (Das et al. 2024), Chronos (Ansari et al. 2024), Lag-Llama (Rasul et al. 2023), MOIRAI (Woo et al. 2024)—has established that large-scale pretraining enables zero-shot forecasting across domains. Yao et al. (2025) found that model architecture significantly influences scaling efficiency. Most relevant to our work, Shi et al. (2024) showed that in time series, additional data does not expand the combinatorial coverage of contexts as directly as additional tokens do in language modeling. Our findings provide a concrete, domain-specific instance of this principle: in building energy, the marginal information content of additional stock-model buildings decreases rapidly because they occupy similar regions of operational schedule space.

### 3.3 Data-Centric AI and Instance Normalization

The data-centric AI perspective (Zha et al. 2023) holds that improving data quality is often more productive than improving model architecture. Our 12-dimensional LHS design applies this principle to the energy domain: rather than sampling from a fixed distribution, we construct a training set that fills the space of plausible operations.

RevIN (Kim et al. 2022) normalizes instance-level load magnitude and variability before the encoder and restores the scale after prediction. Our finding that RevIN's benefit is regime-dependent—helpful for small diverse datasets, harmful at large scale—extends the understanding of when instance normalization helps versus hurts, connecting to broader questions about the role of magnitude information in time series models.

### 3.4 Generalization Theory and Domain Adaptation

Our theoretical framework builds on the classical connection between covering numbers and generalization (Anthony and Bartlett 1999) and domain adaptation theory (Ben-David et al. 2010). Ben-David et al. showed that transfer error depends on the divergence between source and target distributions; we specialize this to the building energy domain by defining divergence in terms of coverage of the operational parameter space. The Lipschitz continuity assumption (A1) is standard in approximation theory and justified by the physics-based nature of EnergyPlus. The key theoretical insight — that coverage, not count, governs generalization — parallels findings in active learning (Settles 2009) and experimental design (Sacks et al. 1989), but to our knowledge has not been applied to zero-shot building load forecasting.

---

## 4. Method
### 4.1 Parametric Building Simulation with LHS-Designed Operational Diversity

Training data are generated through EnergyPlus (Crawley et al. 2001) simulation of commercial buildings with parametrically varied operational schedules. The goal is not to replicate a building stock distribution but to promote broad coverage of the temporal pattern space.

**Building archetypes.** We use 14 archetype models derived from the DOE commercial reference buildings (Deru et al. 2011), adapted to Korean building codes and climate zones. Twelve archetypes correspond directly to DOE reference types: large office, medium office, small office, stand-alone retail, strip mall, primary school, hospital, large hotel, warehouse, midrise apartment, full-service restaurant, and quick-service restaurant. Two additional archetypes—highrise apartment and university—were created by modifying the DOE midrise apartment and primary school models, respectively, with Korean-specific HVAC configurations and occupancy schedules. Four DOE reference types were not used: supermarket and outpatient healthcare were excluded because their refrigeration-dominated or medical-equipment-dominated profiles require specialized schedule parameterization; secondary school was omitted to avoid redundancy with the primary-school-derived education archetypes; and small hotel was omitted because the large hotel archetype already captures the hospitality operating regime. The apartment archetypes are multifamily residential buildings that follow building-level metering and commercial HVAC configurations, distinct from single-family residential load regimes discussed in Section 6.4. Each archetype begins from a DOE reference model geometry with code-compliant envelope thermal properties, HVAC system sizing, and internal load densities.

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
| scale_mult | 0.3–3.0 | Post-simulation amplitude multiplier |
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

### 4.2 Model Architecture

We use a standard encoder-decoder Transformer: 3 encoder and 3 decoder layers, d_model = 512, 8 attention heads, feedforward dimension 1024, totaling 15.8M parameters (Transformer-M). Input consists of 168 hourly load values with temporal features (day-of-year sinusoidal encoding, day-of-week and hour-of-day embeddings). The model predicts 24-hour-ahead Gaussian distributions through autoregressive decoding with Gaussian NLL loss. Using the same Transformer backbone as Emami et al. (2023) minimizes model-capacity confounds in the controlled comparisons.

**Reversible Instance Normalization.** Before encoding, the 168-hour context window is normalized by its instance mean and standard deviation; after decoding, the normalization is reversed. This removes building-specific load magnitude and variability, allowing the model to focus on temporal shape. We apply RevIN symmetrically in all controlled experiments, including stock-model baselines, to ensure fair comparison. The comparison with the original stock-model pipeline (which does not use RevIN) is discussed in Section 5.1.

**No geographic features.** Some zero-shot approaches provide latitude and longitude embeddings as model inputs. We set both to zero for all buildings—both training and evaluation—to test whether operational diversity alone suffices for generalization. The ablation in Section 4.4 confirms that actual coordinates provide no benefit, consistent with RevIN absorbing the scale differences that coordinates might otherwise encode.

### 4.3 Training Protocol

We apply a global Box-Cox transform (Box and Cox 1964) fitted on our simulation data (λ = −0.067). AdamW (Loshchilov and Hutter 2019) with learning rate 6 × 10⁻⁵, weight decay 0.01, cosine annealing, and 500-step warmup. Training runs for 18,000 gradient steps with batch size 128, corresponding to approximately 9 epochs over 700 buildings. Data augmentation includes window jitter (±1–6h), Gaussian noise (σ = 0.02 in Box-Cox space), and amplitude scaling (U[0.85, 1.15]).

With 700 buildings and 18,000 steps, each building is seen roughly 3,300 times across window positions—in contrast to the stock-model baseline, where each of 900K buildings contributes roughly 2–3 windows. Three factors mitigate overfitting despite this repeated exposure: (1) augmentation ensures each exposure presents a different view; (2) RevIN prevents memorizing absolute load levels; (3) the high inter-building diversity from LHS ensures patterns learned from any one building generalize.

### 4.4 Evaluation Protocol

We evaluate on a standardized real-building test set of 955 commercial-labeled load series (Emami et al. 2023): 611 from U.S. commercial buildings in the Building Data Genome Project 2 (BDG-2) (Miller et al. 2020), drawn from four university and government campuses spanning Mediterranean, semi-arid, humid subtropical, and humid continental climates, and 344 from Portuguese electricity consumers (Trindade 2015) (labeled "commercial" in the benchmark metadata, though some may include mixed-use or residential connections), with 15 out-of-vocabulary buildings excluded per the original specification. Per-building NRMSE = sqrt(MSE) / mean(actual), aggregated by median across buildings. The Normalized Continuous Ranked Probability Score (NCRPS) is computed as NCRPS = mean(CRPS) / mean(actual load), which evaluates the quality of the predicted Gaussian distribution. Because our NCRPS implementation normalizes CRPS by mean actual load, whereas the BuildingsBench leaderboard reports RPS under its official evaluation script, we report NCRPS values only for models retrained with our pipeline. The official BB-900K RPS (5.21 on the leaderboard) should not be directly compared with our NCRPS unless evaluated with the same script. The sliding window uses stride 24 hours, context 168 hours, and prediction horizon 24 hours. We reproduced the stock-model baseline using the official checkpoint and obtained 13.27%, confirming pipeline equivalence within 0.01 pp of the reported 13.28%.

---

## 5. Experiments and Results
### 5.1 Equal-Scale Controlled Validation (Prediction 1)

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

The equal-scale comparison provides evidence for the central claim of the proposed methodology: that deliberate design of operational diversity can produce more transferable training data than sampling from a fixed stock-model distribution. Under matched conditions (same architecture, same optimizer, same augmentation, same number of buildings, same RevIN), the LHS-designed corpus outperforms the stock-model sample by **1.33 pp** at seed 42 (12.93% vs. 14.26%). The equal-scale BB-700 comparison uses a single stock-model subset with a single training seed, and therefore does not quantify subset-selection uncertainty. The 1.33 pp advantage should be interpreted as evidence from a controlled but not fully repeated comparison: the gap is substantially larger than the observed inter-seed variation of Korean-700 (±0.17 pp), but a definitive claim would require evaluating multiple BB-700 subsets across multiple seeds. Even without augmentation, Korean-700 (13.67%) outperforms the aug-matched stock-model control by 0.59 pp, and the U.S.-weather variant (13.64%) by 0.62 pp—smaller but directionally consistent advantages. Because RevIN, augmentation, model architecture, and training budget are matched, the gap is unlikely to be explained by these factors. The primary remaining difference is the design and source of the training data, although uncontrolled factors such as building codes, envelope properties, and climate mapping cannot be fully excluded (Section 5.5).

As a secondary observation, the designed dataset achieves performance comparable to the large stock-model reference (five-seed mean 13.11 ± 0.17% vs. 13.27%) without geographic metadata. A paired per-building comparison shows Korean-700 achieves lower error on 680 of 955 buildings (71%); the paired bootstrap 95% CI of the median per-building difference, defined as NRMSE(BB-900K) − NRMSE(K-700), is [0.31, 0.39] pp using the seed-42 K-700 checkpoint (Wilcoxon signed-rank p < 0.001). This per-building paired effect is larger than the 0.16 pp gap between aggregate medians because paired differencing removes inter-building variance.

### 5.2 Weather-Origin Ablation

To test whether the observed benefit depends on the Korean weather files, we retrained on the same 700 LHS-designed buildings using U.S. TMY weather (Seoul → Washington DC, Busan → Atlanta, Daegu → Charlotte, Gangneung → Boston, Jeju → Miami). The five-seed mean of US-TMY-700 is 13.64 ± 0.65%, within 0.53 pp of Korean-700. The U.S.-TMY model still outperforms the equal-scale stock-model control (14.26%) by 0.62 pp, suggesting that schedule design remains beneficial after replacing the Korean weather files with approximately matched U.S. TMY files. The higher seed variance (0.65% vs. 0.17%) indicates that weather choice affects run-to-run stability.

### 5.3 N-Scaling: Evidence for Coverage Saturation (Prediction 2)

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

Under the fixed 18,000-step training budget, performance improves sharply from n = 1 to n = 5 (14.72% to 13.28%), plateauing near the external reference level (13.27%) from n ≈ 5 onward. Beyond this point, gains become incremental—fluctuations within ±0.35 pp rather than clear monotonic improvement. The plateau does not by itself prove that coverage is the operative mechanism — it is also consistent with optimizer saturation under a fixed step budget, or with an architecture capacity ceiling. The observed pattern is consistent with the joint effect of coverage saturation and optimization budget allocation as described in Section 2: with $d = 12$ schedule parameters, the covering radius decreases heuristically as $n^{-1/12}$, so doubling $n$ reduces $\varepsilon^*$ by only ~6%. Once coarse coverage is achieved, additional samples mainly refine existing regions rather than introducing qualitatively new patterns.

The contrast with stock-model scaling is instructive (Appendix B): increasing from 700 to 7,000 stock-model buildings improves NRMSE by only 0.78 pp (15.28% to 14.50%), and even 7,000 stock-model buildings do not approach 700 LHS-designed buildings (13.11%). This comparison is consistent with the interpretation that adding stock-model buildings from a concentrated generation pipeline yields smaller diversity gains than deliberately broadening operational coverage.

### 5.4 Illustrative Coverage Proxy

To connect the conceptual framework with the performance comparisons, we compute an illustrative coverage proxy for the K-700 LHS design in the 12-dimensional normalized parameter space $[0,1]^{12}$. For a point set $S$, the proxy covering radius is estimated as $\varepsilon^*(S) = \max_{x \in T} \min_{s \in S} \|x - s\|_2$, where $T$ is a set of $10^5$ uniformly sampled test points. This analysis is descriptive: it compares LHS with simple synthetic reference samplers, not with the full BuildingsBench generation process.

**Table 4b.** Covering radius comparison in $[0,1]^{12}$ ($n = 50$ points per method, $10^5$ test points).

| Sampling Method | Mean distance | ε* (max) | ε* ratio vs. LHS |
|:----------------|:-------------:|:--------:|:-----------------:|
| K-700 LHS | 0.862 | 1.399 | 1.00 |
| Random Uniform | 0.854 | 1.412 | 1.01 |
| Clustered (stock-model proxy) | 1.089 | 1.985 | 1.42 |

The clustered set simulates a concentrated sampler by drawing 50 points from $\mathcal{N}(\mu_{\text{typical}}, 0.15I)$ clipped to $[0,1]^{12}$, where $\mu_{\text{typical}}$ reflects typical commercial building parameters (e.g., operating start hour ≈ 8, weekday-dominant operation, moderate baseloads). Its proxy covering radius is 42% larger than LHS, meaning that in this illustrative setup the worst-case target point lies farther from the nearest training sample. We treat this result only as intuition for why broader sampling can help; it is not a direct measurement of BuildingsBench coverage.

LHS and random uniform achieve similar covering radii at $n = 50$ because both produce approximately space-filling samples in high dimensions. The advantage of LHS is its guaranteed marginal uniformity: each parameter dimension is covered with exactly $n$ strata, eliminating the possibility of leaving large marginal gaps that random sampling allows with non-negligible probability. The within-set nearest-neighbor CV confirms this: LHS (CV = 0.154) is comparable to uniform (0.145) and substantially more uniform than clustered (0.201).

These proxy measurements are directionally consistent with the mechanism proposed in Section 2: the performance advantage of LHS-designed data may arise from broader operational coverage rather than from larger sample size alone.

#### Feature-Space Proximity Does Not Explain the Gain

To assess how well each training corpus covers the distribution of real buildings, we extract eight time-series features from each building: night/day load ratio, weekend/weekday ratio, daily peak CV, operating hours, morning ramp rate, weekly autocorrelation, seasonality CV, and baseload fraction. For each of the 955 real evaluation buildings, we compute the nearest-neighbor distance to K-700 and BB-700 in this normalized 8D feature space.

**Table 7.** Feature-space proximity to real buildings. Mean NN distance from each of the 955 real evaluation buildings to the nearest point in the training dataset (lower = closer to real). MMD: Maximum Mean Discrepancy with RBF kernel.

| Metric | K-700 | BB-700 |
|:-------|:-----:|:------:|
| Mean NN distance | 0.148 | 0.109 |
| P95 NN distance | 0.413 | 0.251 |
| Fraction within d < 0.2 | 78.9% | 92.0% |
| MMD to Real (RBF) | 0.609 | 0.498 |
| Per-feature overlap (mean) | 53.7% | 82.1% |

BB-700 lies closer to real buildings in feature space than K-700 across all metrics. Per-feature analysis confirms this: BB-700 covers 76% of the real operating-hours range and 84% of the weekly-autocorrelation range, whereas K-700 covers only 42% and 23%, respectively. The narrow K-700 operating-hours range reflects high baseload and nighttime equipment fractions in many LHS samples, which cause the feature extractor to classify them as near-continuous operation even when the `op_duration` parameter spans 8–24 h.

This result clarifies what "coverage" means in the context of zero-shot transfer. K-700 achieves superior forecasting accuracy despite covering a narrower region of the observed feature distribution. The benefit of LHS is therefore not explained by closer marginal feature matching to the real-building distribution. Rather, LHS acts as a structured intervention over operational factors, creating disentangled temporal-pattern variation that teaches the model robust pattern decomposition across the designed operational parameter space — including extreme combinations that rarely occur in real stocks. RevIN then bridges the remaining distributional gap at inference time by removing building-specific magnitude and variability.

### 5.5 Ablation Studies (Prediction 3)

Table 5 reports ablations for RevIN and geographic metadata, using seed-42 for consistent comparison.

**Table 5.** Ablation results (seed 42, checkpoint selected on held-out simulation validation set, evaluated on the 955 real-building test set. The validation set consists of withheld simulation buildings and does not overlap with the 955-building evaluation set).

| Experiment | NRMSE (%) | Delta | Interpretation |
|------------|:----------:|:-----:|----------------|
| K-700 RevIN ON (baseline) | 12.93 | — | Seed 42, lat/lon = zero |
| K-700 RevIN OFF | 14.81 | +1.88 | RevIN contributes 1.88 pp (seed 42); 1.6 pp across multi-seed means (Appendix A) |
| Actual lat/lon coordinates | 12.93 | 0.00 | Geographic metadata provides no benefit |

Exploratory runs from an earlier pipeline iteration are reported in Appendix C for transparency. These runs—using stock-model-fitted Box-Cox parameters, extended training, scaling to 70K buildings, and seasonal decomposition before RevIN—all degraded performance, but because they were conducted under a different pipeline configuration, they are not used as primary evidence for the claims in this paper.

---

## 6. Discussion
### 6.1 Mechanistic Interpretation of RevIN's Regime-Dependent Effect (Prediction 3)

The asymmetry in RevIN's effect across dataset scales (Fig. 4) admits a coherent mechanistic interpretation that extends beyond our specific setting.

RevIN normalizes each context window to zero mean and unit variance, stripping absolute load magnitude. With a small LHS-designed dataset, the model has not seen enough magnitude variation to internalize it; RevIN reduces this burden by normalizing instance-level scale and variability, yielding a substantial improvement (Table 5). With a large stock-model corpus—where each building contributes only a few training windows—magnitude may carry transferable information because it is correlated with building type, HVAC configuration, and schedule assumptions in the generation pipeline. Removing this information may discard something useful, producing a measurable degradation (Table 3).

A structural factor reinforces this. In stock-model corpora, magnitude is informative about building type because HVAC, envelope, and schedule are jointly determined by the generation pipeline—magnitude and temporal shape are correlated. In our LHS design, magnitude reflects only the scale_mult parameter, sampled independently of schedule parameters. RevIN's removal of magnitude therefore discards less useful information in LHS data than in stock-model data.

This finding has implications for the broader time series community: RevIN's benefit depends not just on dataset size but on the information structure of the training data—specifically, whether magnitude carries predictive information about temporal dynamics.

### 6.2 Design Principles Behind Transferable Pattern Coverage

The methodology's effectiveness can be traced to three design principles that together produce a training corpus with high transferability per building.

1. **Independent marginal coverage through LHS.** Each of the 12 schedule parameters is sampled with uniform marginal coverage, ensuring that the training set visits the full range of each operational dimension—including extreme combinations (e.g., high baseload with short operating hours) that rarely occur in real building stocks but expose the model to temporal dynamics it must handle at inference time. This helps explain why 700 LHS-designed buildings outperform 700 stock-model buildings under identical conditions: LHS spreads samples across the operational schedule space rather than concentrating them around its mode.

2. **Archetype-stratified diversity.** Stratifying the 12D LHS across 14 building archetypes ensures that schedule diversity interacts with distinct thermal mass, HVAC response, and internal gain profiles. This produces a combinatorial expansion of temporal patterns: the same nighttime equipment retention fraction yields qualitatively different load shapes in a hospital versus a strip mall. The n-scaling plateau at a modest number of samples per archetype (Table 4) indicates that this archetype × schedule interaction space is covered with relatively few samples.

3. **Decoupled magnitude and temporal shape.** The scale_mult parameter is sampled independently of the 11 schedule parameters, reducing the extent to which load magnitude encodes schedule-driven temporal dynamics. This structural property helps explain why RevIN—which strips magnitude—helps in our design but hurts in stock-model corpora, where magnitude and schedule are jointly determined by the generation pipeline and therefore correlated.

These principles also explain why the sub-epoch training regime observed in large-scale stock-model training (Emami et al. 2023) is consistent with our hypothesis: the distinct temporal patterns in a stock-model corpus are learned quickly because many buildings produce similar patterns, and further training memorizes pipeline-specific artifacts rather than learning new dynamics.

From a scaling-law perspective, this aligns with Shi et al. (2024): in building load time series, the marginal information content of additional buildings decreases rapidly when the operational diversity of the corpus is fixed. Our methodology addresses this directly by promoting broad diversity at design time.

### 6.3 Practical Implications

The reframing from scale to design has immediate practical consequences. An organization can generate several hundred parametric EnergyPlus simulations conforming to local building codes and climate, train a standard Transformer with RevIN and augmentation, and deploy zero-shot forecasting—without assembling a national building stock database.

The computational cost is modest: in our implementation, 700 simulations completed in approximately 4 hours on a single workstation (10-core CPU, 64 GB RAM); training takes approximately 2 hours on a single consumer GPU (NVIDIA RTX 4090). The entire pipeline from simulation to deployed model takes less than a day. No location metadata, building-type classification, or HVAC information is required at inference time—only 168 hours of historical load data.

### 6.4 Limitations (Including Prediction 4)

Our parametric simulations differ from the stock-model baseline in schedule design, climate, building codes, and envelope parameters. The equal-scale controlled experiment (Section 4.1) controls for augmentation, model architecture, and training budget, but cannot fully disentangle all data-source-related confounds. The U.S.-TMY ablation substantially reduces but does not eliminate the climate-origin confound.

The augmentation applied to our method (window jitter, Gaussian noise, amplitude scaling) was not part of the original stock-model training pipeline. The aug-matched control addresses this asymmetry at equal scale, but the comparison with the original baseline (which uses no augmentation) includes this confound. Even without augmentation, our method outperforms the aug-matched stock-model control (Table 3), but falls above the full-scale baseline.

The equal-scale BB-700 comparison uses a single stock-model subset with a single training seed. Repeating this comparison across multiple subsets and seeds would strengthen the claim but requires approximately 50 hours of additional training; we leave this for future work. The 1.33 pp advantage over BB-700 should therefore be interpreted cautiously.

The objective of this study is not to claim a large performance breakthrough, but to demonstrate that a systematically designed simulation dataset can provide competitive zero-shot transfer with substantially lower data requirements. The margin between the designed dataset and the large-scale reference is comparable to inter-seed variation (Table 3). The paired per-building comparison confirms statistical significance (Section 4.1), but the margin is small. The contribution is methodological: showing that systematic simulation design can substantially reduce reliance on brute-force dataset scaling in this benchmark setting.

All experiments use a single model architecture (Transformer-M); generalization to PatchTST, MOIRAI, or other architectures has not been tested.

The coverage-based framework in Section 2 is deliberately simplified. It is useful for interpreting the results, but it does not estimate the true target distribution, prove asymptotic rates for the actual benchmark pipeline, or directly measure the coverage of BuildingsBench itself.

On residential buildings, our model yields accuracy comparable to the Persistence Ensemble. The framework of Section 2 suggests that this is a coverage gap: residential buildings occupy a parameter space $\Theta_{\text{res}}$ that is largely disjoint from the commercial space $\Theta_{\text{com}}$ used for training — residential schedules are driven by occupant behavior rather than institutional operations, with fundamentally different baseload ratios, operating hours, and weekly patterns. In practical terms, the current design does not cover those regimes. Extending the approach to residential buildings therefore requires a separate $\Theta_{\text{res}}$ with residential-specific schedule parameters.

Real-world validation beyond the evaluation set used in this study has not been conducted; broader testing across diverse building types in multiple countries is needed.

---

## 7. Conclusion
We presented a coverage-based interpretation of the Operational Diversity Hypothesis for zero-shot building load forecasting. The central claim is practical rather than absolute: a compact simulation dataset can remain competitive when it is designed to span diverse operating regimes instead of merely increasing sample count. Controlled experiments are broadly consistent with this interpretation:

(1) in the single-subset equal-scale control, the LHS-designed dataset outperforms the BB-700 stock-model sample (12.93% vs. 14.26% NRMSE) under matched training conditions; (2) N-scaling plateaus around $n \approx 5$ per archetype under the fixed training budget used here; (3) RevIN reduces NRMSE by 1.88 pp on LHS data but increases it by 0.62 pp on stock-model data, consistent with differences in how magnitude information is structured; and (4) residential forecasting remains weak when training covers only commercial operating regimes.

Taken together, the results suggest that the data requirement for zero-shot building energy forecasting should be framed partly as a structured parameter-intervention problem, not only as a scale problem. Feature-space analysis (Table 7) confirms that the benefit does not arise from closer distributional matching to real buildings; rather, LHS acts as a disentangled intervention over operational factors that teaches robust pattern decomposition. A dataset of 700 simulations designed for broad operational-factor variation achieves transfer performance comparable to a 900K-building stock-model corpus, reducing the entire pipeline to less than a day on a single workstation.

The principal open questions are extending the framework to residential buildings, testing generalization across model architectures (PatchTST, MOIRAI), measuring coverage more directly against real and stock-model datasets, and resolving the augmentation asymmetry between training pipelines.

---

## References

Anthony M, Bartlett PL (1999). Neural Network Learning: Theoretical Foundations. Cambridge University Press.

Amasyali K, El-Gohary NM (2018). A review of data-driven building energy consumption prediction studies. *Renewable and Sustainable Energy Reviews*, 81: 1192–1205.

Ansari AF, et al. (2024). Chronos: Learning the Language of Time Series. *Transactions on Machine Learning Research*.

Ben-David S, Blitzer J, Crammer K, Kuber A, Pereira F, Vaughan JW (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2): 151–175.

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

Sacks J, Welch WJ, Mitchell TJ, Wynn HP (1989). Design and Analysis of Computer Experiments. *Statistical Science*, 4(4): 409–423.

Settles B (2009). Active Learning Literature Survey. Computer Sciences Technical Report 1648, University of Wisconsin-Madison.

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

## Appendix C: Exploratory Ablation Results (Earlier Pipeline)

Exploratory ablations from an earlier pipeline iteration, reported for transparency. These results are not directly comparable to the final pipeline and are not used as primary evidence. Baseline is K-700 at 12.93%. Δ is NRMSE increase in percentage points.

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
