# Seven Hundred Simulations Suffice: Matching BuildingsBench through Operational Diversity in Zero-Shot Load Forecasting

**Author**: Jeong-Uk Kim
**ORCID**: 0000-0002-3839-3845
**Affiliation**: Department of Electrical Engineering, Sangmyung University, Seoul 03016, South Korea
**E-mail**: jukim@smu.ac.kr

---

## Abstract

Zero-shot building load forecasting—predicting energy consumption for unseen buildings without retraining—is essential for grid balancing and demand response. BuildingsBench assembled 900,000 U.S. building simulations and achieved 13.28% NRMSE on 955 real U.S. commercial buildings and Portuguese electricity consumers, establishing data scale as the prevailing paradigm. We challenge this with 700 Korean-weather EnergyPlus simulations—fifty per archetype across 14 building types—with operational schedules sampled via 12-dimensional Latin Hypercube Sampling (LHS), using Reversible Instance Normalization (RevIN) and no geographic features. This small corpus achieves 13.11 ± 0.17% NRMSE (five-seed mean) on the same evaluation set, outperforming the 900,000-building state-of-the-art (SOTA, 13.28%) on the five-seed mean with the best seed reaching 12.93%. Controlled experiments isolating the design effect—comparing 700 LHS-designed buildings against 700 BuildingsBench-sampled buildings—show a persistent 1.15 pp advantage with matched augmentation (1.61 pp without), supporting operational diversity as a primary driver beyond RevIN or augmentation alone. Applying RevIN to the full 900K corpus degrades performance by 0.62 pp, indicating that RevIN benefits small diverse datasets but hurts large homogeneous corpora. N-scaling saturates rapidly: 70 buildings already match the SOTA (13.28 ± 0.12%). A U.S.-weather ablation—retraining on the same 700 LHS schedules with U.S. TMY weather—yields 13.64 ± 0.65%, outperforming BB-700 (14.26%) while remaining more variable than Korean-700, supporting operational diversity as a major driver while revealing residual weather sensitivity.

**Keywords**: building energy forecasting, zero-shot learning, foundation models, parametric simulation, data-centric AI, reversible instance normalization

**Highlights**:
- 700 diverse simulations achieve SOTA-level accuracy at 0.08% of training volume
- 12-dimensional Latin Hypercube Sampling generates operationally diverse training data
- RevIN helps small diverse data but degrades performance on large homogeneous corpora
- U.S.-weather ablation supports operational diversity with residual climate effect

---

## 1. Introduction

Short-term load forecasting underpins grid balancing, demand response, and real-time building energy management [1, 2]. The classical approach trains a separate model for each building, which works well when historical data are available but is inapplicable to newly instrumented buildings or those with limited historical data. Foundation models for time series—trained once and transferred across unseen targets—offer an alternative: learn general temporal patterns from large corpora and predict unseen buildings without fine-tuning.

BuildingsBench [3] operationalized this idea for the building energy domain. By assembling Buildings-900K, a synthetic corpus of roughly 900,000 residential and commercial buildings drawn from the NREL End-Use Load Profiles database [4], and training an encoder-decoder Transformer with Gaussian negative log-likelihood loss, the authors achieved 13.28% median NRMSE on 955 unseen real load series from U.S. commercial buildings and Portuguese electricity consumers (which we reproduce at 13.27%). The result established a benchmark and carried an implicit message: scale matters. More synthetic buildings, more patterns, better generalization.

Recent work in the broader time series community has begun to challenge this framing. Shi et al. [5] showed that scaling laws for time series forecasting diverge from those in language modeling, finding that more capable models do not always outperform less capable ones, and that performance depends not only on model and data scale but also on the look-back horizon, which interacts with autocorrelation, periodicity, and non-stationarity in the data. MOIRAI-MoE [6] demonstrated a sparse mixture-of-experts architecture with automatic token-level specialization, achieving superior zero-shot performance without increasing model or data scale. These findings suggest that brute-force scaling may not be sufficient for generalization. The data-centric AI community [7] offers a further angle: rather than collecting more data from the same distribution, designing better data may be more productive.

We test this hypothesis directly in the building load forecasting setting. Our starting observation is that Buildings-900K, despite its size, draws from a single national stock-model generation pipeline—the NREL End-Use Load Profiles for the U.S. building stock. Importantly, this is a single-country, single-pipeline corpus: all 900,000 buildings are U.S. simulations. Adding more buildings from the same generation pipeline increases volume but not necessarily the diversity of temporal patterns a model encounters. An alternative is to construct a small training set with maximal operational diversity by design.

We generate 700 EnergyPlus simulations (50 per archetype across 14 building types) with operational schedules sampled via 12-dimensional Latin Hypercube Sampling. The 12 parameters—covering operating hours, baseload levels, weekend patterns, ramp characteristics, equipment retention, and seasonal variation—are designed to span a broad space of plausible commercial building operations, beyond what a random stock-model sample of comparable size is likely to cover. Combined with Reversible Instance Normalization (RevIN) [8], which absorbs building-specific load magnitude and variability at inference time, this small but diverse training set achieves 13.11 ± 0.17% median NRMSE across five seeds on the BuildingsBench evaluation protocol, with the best seed reaching 12.93%. The model uses no geographic features (latitude and longitude are set to zero).

Controlled experiments on matched 700-building subsets—one from our parametric simulations, one from BuildingsBench—show that the improvement cannot be attributed to RevIN alone and is strongly linked to data design. Applying RevIN to the full 900K BuildingsBench corpus worsens performance from 13.27% to 13.89%, an asymmetry suggesting that RevIN is beneficial in small-data regimes but not universally helpful at scale. The n-scaling curve shows that 70 buildings already match the SOTA at 13.28 ± 0.12%. A U.S.-weather/LHS ablation—retraining on the same 700 LHS schedules with U.S. TMY weather—yields a five-seed mean of 13.64 ± 0.65%, still outperforming the matched BB-700 control (14.26%), supporting operational diversity as a major driver while revealing residual weather sensitivity.

The comparison is not free of confounds: our simulations differ from BuildingsBench in building-code assumptions and augmentation strategy, factors that controlled experiments partially address but cannot fully disentangle. A five-seed U.S.-TMY/LHS ablation substantially reduces the climate-origin component of the confound (Section 4.2). Remaining limitations are discussed in Section 5.

---

## 2. Related Work

### 2.1 Building Energy Forecasting

Prior methods ranging from statistical models [9] to gradient boosting [10], recurrent networks [11], and Transformers [12, 13] improved per-building accuracy but remained building-specific. Transfer learning through domain adaptation [14] and fine-tuning [15] relaxes this constraint but still requires target-domain data. Our work eliminates the need for target-building fine-tuning through zero-shot inference.

### 2.2 BuildingsBench and Zero-Shot Load Forecasting

BuildingsBench [3] framed building load forecasting as a zero-shot generalization problem. The training corpus, Buildings-900K, comprises approximately 350,000 commercial (ComStock) and 550,000 residential (ResStock) synthetic buildings with hourly electricity consumption generated by the NREL EULP pipeline [4]. The model—an encoder-decoder Transformer predicting 24-hour Gaussian distributions from 168-hour context—trains for a fraction of one epoch (approximately 0.067 epochs by our calculation, equivalent to roughly 2–3 sampled training windows per building on average) using a fitted global Box-Cox normalization (λ = −0.067, as measured from the provided checkpoint). The reported state-of-the-art (SOTA) is 13.28% median NRMSE on commercial buildings. Our work keeps the model architecture and evaluation protocol fixed while changing the training data and normalization strategy.

The sub-epoch training regime is consequential. BuildingsBench found that training for less than one full pass over the data produced the best generalization to real buildings; additional training degraded OOD evaluation performance. The authors interpreted this as evidence that the model memorizes synthetic patterns faster than it learns transferable representations, making training duration as consequential as data volume.

### 2.3 Scaling Laws and Their Limits in Time Series

The proliferation of general-purpose time series foundation models—TimesFM [16], Chronos [17], Lag-Llama [18], MOIRAI [19]—has established that large-scale pretraining enables zero-shot forecasting across domains. These models train on millions to billions of time series from heterogeneous sources. Shi et al. [5] investigated whether the power-law scaling observed in language models transfers to time series and found that it does not: more capable models do not always outperform less capable ones, and the look-back horizon plays a distinct role in time series, interacting with autocorrelation, periodicity, and non-stationarity in ways not captured by standard language-model scaling theory. Yao et al. [20] found that model architecture significantly influences scaling efficiency, with encoder-only Transformers showing slightly better parameter scalability and in-distribution performance than decoder-only alternatives. Most relevant to our work, MOIRAI-MoE [6] showed that a mixture-of-experts architecture can outperform its monolithic predecessor through token-level specialization rather than increased model or data scale. The relationship between data volume and generalization in time series is more complex than in language modeling, where additional tokens more directly expand the combinatorial coverage of linguistic contexts.

### 2.4 Data-Centric AI and Reversible Instance Normalization

The data-centric AI perspective [7] holds that improving data quality is often more productive than improving model architecture. Our 12-dimensional LHS design applies this principle to the energy domain: rather than sampling from a fixed distribution, we construct a training set that fills the space of plausible operations.

RevIN [8] complements this approach by normalizing instance-level load magnitude and variability before the encoder and restoring the scale after prediction, allowing the model to focus on temporal shape. Its interaction with dataset scale is examined in Section 4.

---

## 3. Method

### 3.1 Parametric Building Simulation

Training data are generated through EnergyPlus [21] simulation of Korean commercial buildings with parametrically varied operational schedules.

**Building archetypes.** We use 14 building archetypes based on DOE commercial reference building models [22] and adapted to Korean building codes and climate zones: office, retail, school, hotel, hospital, apartment (midrise and highrise), small office, large office, warehouse, strip mall, restaurant (full-service and quick-service), and university. The apartment (highrise) and university archetypes extend the standard DOE set following PNNL prototype conventions; the remaining 12 map directly to DOE reference types. Each archetype begins from a DOE reference model with Korean-code-compliant envelope thermal properties, HVAC system sizing, and internal load densities.

**Climate coverage.** Simulations span five Korean cities across three climate zones: Seoul (central), Busan (southern), Daegu (southern), Gangneung (central-coastal), and Jeju (subtropical). Typical Meteorological Year (TMY) weather files for each city drive the simulations. For the U.S.-TMY/LHS ablation (Section 4.2), the same 700 schedule samples and archetypes were rerun with U.S. TMY files mapped by approximate ASHRAE climate-zone similarity: Seoul → Washington DC (4A), Busan → Atlanta (3A), Daegu → Charlotte (3A), Gangneung → Boston (5A), and Jeju → Miami (1A).

**Operational parameter space.** Building operational schedules are parameterized by 12 continuous variables sampled via Latin Hypercube Sampling [23] (Table 1). These parameters govern operating hours (op_start, op_duration), baseload and ramp characteristics (baseload_pct, ramp_hours), occupancy-related patterns (weekend_factor, equip_always_on, night_equipment_frac), and stochastic variation (daily_noise_std, weekly_break_prob, seasonal_amplitude, process_load_frac, scale_mult).

**Table 1.** Twelve-dimensional LHS parameter space for operational schedule generation.

| Parameter | Range | Description |
|-----------|-------|-------------|
| op_start | 0--12 h | Operation start time |
| op_duration | 8--24 h | Operating hours per day |
| baseload_pct | 25--98% | Off-hours load as fraction of peak |
| weekend_factor | 0--1.2 | Weekend-to-weekday load ratio |
| ramp_hours | 0.5--4 h | Transition ramp duration |
| equip_always_on | 30--95% | Always-on equipment fraction |
| daily_noise_std | 5--35% | Day-to-day stochastic variation |
| scale_mult | 0.3--3.0 | Load density multiplier |
| night_equipment_frac | 30--95% | Nighttime equipment retention |
| weekly_break_prob | 0--25% | Weekly pattern disruption probability |
| seasonal_amplitude | 0--30% | Seasonal load oscillation amplitude |
| process_load_frac | 0--50% | Constant process load fraction |

The 12 parameters were selected to span the primary operational degrees of freedom identified in commercial building energy audits; sensitivity to individual parameters has not been analyzed. LHS [23] ensures stratified marginal coverage across all 12 dimensions, improving space filling relative to random sampling, and produces buildings that range from 24/7 high-baseload facilities (resembling data centers or hospitals) to weekday-only offices with steep morning ramps and low overnight loads. The overall pipeline from archetype selection through simulation to zero-shot inference is illustrated in Fig. 1. For the n = 50 configuration used in our main experiments, this yields 50 samples per archetype × 14 archetypes = 700 buildings. Each sample modifies the EnergyPlus IDF file—adjusting internal load schedules, equipment power densities, and occupancy patterns—runs a full annual simulation (8,760 hours), and extracts hourly total electricity consumption.

Unlike stock-model sampling—which reflects the statistical distribution of real buildings (many near typical conditions, few at extremes)—LHS ensures uniform marginal coverage in each dimension. Stock-model sampling asks "what does the building stock look like?"; LHS asks "what temporal patterns can buildings produce?"

**Statistical calibration.** Four parameters—night_equipment_frac, weekly_break_prob, seasonal_amplitude, and process_load_frac—represent common operational phenomena in commercial buildings: nighttime equipment retention, weekly schedule disruptions, seasonal load variation, and constant process loads. Table 2 provides a post-hoc sanity check showing that including them brings simulated profiles closer to the aggregate distributional statistics of real buildings. Because these statistics are drawn from the public evaluation corpus, this comparison is best interpreted as distributional calibration rather than a fully blind design step; however, these parameters affect only the marginal distribution of temporal shapes in the training corpus, not any individual building's load values.

**Table 2.** Statistical calibration of simulated vs. real building load profiles, with and without the four diversity-enhancing parameters. Metrics characterize input sequence shape (X), not prediction targets (y). Night/day ratio: nighttime-to-daytime mean load; Autocorr 168h: weekly periodicity; Baseload P95: minimum/maximum daily load; CV P5: day-to-day variability.

| Metric | Without 4 params | With 4 params | BB Real Buildings |
|--------|:----------------:|:-------------:|:-----------------:|
| Night/Day ratio | 0.512 | 0.852 | 0.803 |
| Autocorrelation (168h) | 0.938 | 0.784 | 0.751 |
| Baseload P95 | 0.51 | 0.905 | 0.725 |
| CV P5 (flatness) | 0.21 | 0.032 | 0.105 |

### 3.2 Model Architecture

To isolate the effect of training data from model design, we adopt the identical Transformer architecture used in BuildingsBench [3]. The encoder-decoder model has 3 encoder and 3 decoder layers, d_model = 512, 8 attention heads, and a feedforward dimension of 1024, totaling 15.8M parameters (identical to BuildingsBench Transformer-M). Input consists of 168 hourly load values along with temporal features (day-of-year sinusoidal encoding, day-of-week embedding, and hour-of-day embedding). The model predicts 24-hour-ahead Gaussian distribution parameters (mu, sigma) through autoregressive decoding, trained with Gaussian negative log-likelihood loss.

**Reversible Instance Normalization.** Before encoding, the 168-hour context window is normalized by its instance mean and standard deviation:

$$x_{\text{norm}} = (x - \mu_{\text{ctx}}) / \sigma_{\text{ctx}}$$

After decoding, the normalization is reversed:

$$\hat{y} = \hat{y}_{\text{norm}} \cdot \sigma_{\text{ctx}} + \mu_{\text{ctx}}$$

This removes building-specific load magnitude and variability from the learning problem, allowing the model to focus on temporal shape. RevIN is a standard technique [8] that does not modify the Transformer architecture; it operates as a pre- and post-processing layer. We apply it symmetrically in all controlled experiments, including the BuildingsBench baselines, to ensure fair comparison within our experimental framework. The comparison between our RevIN-equipped model and the original BuildingsBench (which does not use RevIN) is not strictly equivalent; we discuss this asymmetry in Section 5.1.

**No geographic features.** BuildingsBench provides latitude and longitude embeddings as model inputs. We set both to zero for all buildings—both training and evaluation. The ablation in Section 4.4 replaces zero with actual Korean city coordinates for training buildings (Seoul, Busan, Daegu, Gangneung, Jeju), while evaluation buildings retain zero. This produces results indistinguishable from the zero-coordinate baseline when RevIN is active, suggesting that explicit geographic coordinates add little information once instance-level scale differences are normalized by RevIN.

### 3.3 Training Protocol

We apply a global Box-Cox power transform [24] fitted on our Korean simulation data to normalize load values before training. The fitted power parameter λ = −0.067 matches the value we measured from the BuildingsBench checkpoint, but the scale and location parameters (mean_, scale_) differ because they are fitted to the Korean corpus rather than the U.S. stock-model corpus. The optimizer is AdamW [25] with a learning rate of 6 × 10⁻⁵, weight decay of 0.01, cosine annealing schedule, and a 500-step linear warmup. Training runs for 18,000 gradient steps with batch size 128, corresponding to approximately 9 epochs over the 250,600 sliding windows derived from 700 buildings (358 windows per building at stride 24). Data augmentation includes window jitter (±1–6 hours random shift of the context window start), Gaussian noise (sigma = 0.02 in Box-Cox space), and amplitude scaling (uniform in [0.85, 1.15]). Mixed-precision (FP16) autocast is used during training and inference.

Both our training and BuildingsBench use approximately 18,000 gradient steps. BuildingsBench processes 0.067 epochs over 900,000 buildings; we process approximately 9 epochs over 700 buildings. The total number of gradient updates is similar, but distributed very differently: each of our 700 buildings is seen roughly 3,300 times (across different window positions), while each BuildingsBench building contributes roughly 2–3 sampled training windows on average (18,000 × 128 / 900,000 ≈ 2.6). Unlike BuildingsBench's sub-epoch regime, our small corpus is revisited multiple times. Three factors help mitigate overfitting from this high per-building exposure: (1) data augmentation (window jitter, Gaussian noise, and amplitude scaling) ensures that each exposure presents a different view of the training signal; (2) RevIN removes per-building magnitude information, preventing the model from memorizing absolute load levels; and (3) the high inter-building diversity of the LHS design ensures that patterns learned from any one building are also present, in varied form, in many others. The augmentation applied here was not included in the original BuildingsBench training pipeline; this asymmetry is discussed as a limitation in Section 5.4.

### 3.4 Evaluation Protocol

We follow the BuildingsBench test set, sliding-window construction, and aggregation metric without modification. The test set comprises 955 real load series: 611 from U.S. commercial buildings in the Building Data Genome Project 2 (BDG-2) [26], drawn from four university and government campuses (University of California Berkeley, Arizona State University, University of Central Florida, and Washington D.C.), and 344 from Portuguese electricity consumers in the Electricity dataset [27], with 15 out-of-vocabulary buildings excluded per the original specification. The evaluation set spans four distinct U.S. climate zones (Mediterranean, semi-arid desert, humid subtropical, and humid continental) plus Southern Europe, meaning our Korean-trained model is assessed against real load series climatically and geographically far from its Korean training distribution. Per-building NRMSE is computed as sqrt(MSE) / mean(actual) and aggregated by taking the median across buildings. We use the term NRMSE (Normalized Root Mean Squared Error) to match BuildingsBench's convention; this is numerically identical to the Coefficient of Variation of RMSE (CVRMSE) when load values are non-negative, as is always the case for electricity consumption. The sliding window uses stride 24 hours, context 168 hours, and prediction horizon 24 hours.

We reproduced the BuildingsBench SOTA-M result using the official checkpoint (torch 2.0.1, identical Box-Cox parameters) and obtained 13.27% NRMSE, confirming that our evaluation pipeline matches the original within 0.01 percentage points of the reported 13.28%.

---

## 4. Experiments and Results

### 4.1 Main Results

Table 3 presents the primary comparison (visualized in Fig. 2). All models use the same 15.8M-parameter Transformer-M architecture and are evaluated on the identical 955-building test set with the same NRMSE computation.

**Table 3.** Main results on the 955-building evaluation set (U.S. commercial buildings and Portuguese electricity consumers). Korean-700 and US-TMY-700 (RevIN ON, aug) report five-seed mean ± std; RevIN OFF reports three-seed mean ± std; all other rows use seed = 42. NCRPS (lower is better) measures probabilistic calibration in kWh scale. †BB SOTA-M NCRPS unavailable due to Box-Cox protocol mismatch. BB-700 (aug-matched) is trained with identical optimizer, augmentation, and steps as Korean-700 to isolate the data-source effect.

| Model | Training Data | N Buildings | RevIN | Aug | NRMSE (%) | NCRPS (%) | vs SOTA |
|-------|--------------|:-----------:|:-----:|:---:|:----------:|:---------:|:-------:|
| BB SOTA-M (reproduced) | BB 900K | 900,000 | OFF | OFF | 13.27 | —† | — |
| BB 900K + RevIN | BB 900K | 900,000 | ON | OFF | 13.89 | 7.76 | +0.62 |
| **Korean-700 (ours)** | **Korean sim** | **700** | **ON** | **ON** | **13.11 ± 0.17** | **7.14 ± 0.03** | **−0.16** |
| Korean-700 (no aug) | Korean sim | 700 | ON | OFF | 13.67 | 8.16 | +0.40 |
| Korean-700 | Korean sim | 700 | OFF | ON | 14.72 ± 0.28 | 8.29 | +1.45 |
| BB-700 (aug-matched) | BB subset | 700 | ON | ON | 14.26 | 7.80 | +0.99 |
| US-TMY-700 (ours) | U.S. TMY sim | 700 | ON | ON | 13.64 ± 0.65 | 7.53 ± 0.41 | +0.37 |
| BB-700 (no aug) | BB subset | 700 | ON | OFF | 15.28 | — | +2.01 |
| BB-700 | BB subset | 700 | OFF | OFF | 16.44 | — | +3.17 |
| Persistence Ensemble | — | — | — | — | 16.68 | — | +3.41 |

The five-seed mean of Korean-700 with RevIN and augmentation is 13.11 ± 0.17%, surpassing the BB SOTA (13.27%) by 0.16 pp; the best seed reaches 12.93% (0.34 pp improvement). To our knowledge, no prior published work has demonstrated performance in this range using fewer than 1,000 training buildings on the BuildingsBench evaluation protocol. Korean-700 without augmentation (RevIN ON, three-seed mean) achieves 13.67%—0.40 pp above the SOTA—showing that augmentation contributes 0.56 pp and is necessary to cross the SOTA threshold. Without RevIN, performance degrades further to 14.72 ± 0.28% (1.45 pp above the SOTA), though still well below the Persistence Ensemble baseline (16.68%).

The BB-700 aug-matched control—the same architecture trained on 700 randomly sampled BuildingsBench buildings with matched augmentation—achieves 14.26% with RevIN, indicating that the data source matters independently of the normalization strategy. The BB-700 no-aug RevIN ON result of 15.28% and the BB-700 RevIN OFF no-aug result of 16.44% complete the no-augmentation branch of the factorial (an aug-matched BB-700 RevIN OFF experiment was not conducted; see Section 4.2). Augmentation contributes 1.02 pp to BB data (15.28% to 14.26%), yet even the aug-matched BB-700 trails Korean-700 by 1.15 pp—a gap that cannot be explained by augmentation alone.

Adding RevIN to the full BuildingsBench training pipeline degrades performance from 13.27% to 13.89%, a 0.62 percentage-point increase. RevIN helps small data (Korean-700: 14.72% three-seed mean without RevIN to 13.11% five-seed mean with RevIN, a 1.61 pp difference) but hurts large data (BB 900K: 13.27% to 13.89%, a 0.62 pp degradation). For the BB 900K + RevIN experiment, the only modification to the original BuildingsBench pipeline was adding RevIN pre/post-processing (see Appendix A.4). The mechanism behind this asymmetry is examined in Section 5.1.

### 4.2 Decomposing the Improvement

The augmentation-controlled factorial (Korean vs. BB data, RevIN ON, both at n = 700) yields two clean comparisons. With matched augmentation: Korean-700 (13.11%) outperforms BB-700 (14.26%) by 1.15 pp. Without augmentation: Korean-700 (13.67%) outperforms BB-700 (15.28%) by 1.61 pp. Both comparisons use RevIN ON, isolating the data-source effect. The data-source advantage is therefore 1.15–1.61 pp, with augmentation contributing an additional 0.56 pp to the Korean-700 result (13.67% to 13.11%) and 1.02 pp to BB-700 (15.28% to 14.26%).

The RevIN effect is 1.61 pp on Korean data (14.72% three-seed mean to 13.11% five-seed mean) and 1.16 pp on BB-700 (16.44% to 15.28%). An aug-matched BB-700 RevIN OFF experiment was not conducted; if augmentation provides a similar 1.02 pp benefit to RevIN OFF models (as it does to RevIN ON models), the estimated aug-corrected RevIN OFF data-source gap would be approximately 0.70 pp—still meaningful, though this estimate assumes augmentation effects are independent of the RevIN condition. The primary evidence for data-source advantage rests on the no-aug RevIN ON comparison (1.61 pp) and the aug-matched RevIN ON comparison (1.15 pp), both pointing consistently to data design as a major contributor.

To isolate schedule diversity from climate effects, we retrained on the same 700 LHS-designed buildings using U.S. TMY weather files (Seoul → Washington DC, Busan → Atlanta, Daegu → Charlotte, Gangneung → Boston, Jeju → Miami; ASHRAE climate zones matched where possible). The five-seed mean of the U.S.-TMY-700 model is 13.64 ± 0.65% (seeds 42–46: 13.15%, 13.35%, 13.22%, 14.72%, 13.78%), within 0.53 pp of Korean-700 (13.11 ± 0.17%). The best seed (13.15%) falls within Korean-700 seed variability, while the five-seed mean is 0.37 pp above the BB SOTA. The higher variance (0.65% vs. 0.17%) suggests that U.S. weather introduces additional seed sensitivity. Nevertheless, the U.S.-TMY model still outperforms the matched BB-700 baseline (14.26%) by 0.62 pp, indicating that LHS-designed schedule diversity contributes substantially beyond climate origin alone, while weather choice still affects transfer stability.

### 4.3 N-Scaling Analysis

Table 4 and Fig. 3 show how NRMSE varies with the number of buildings per archetype (n), where total buildings = 14n. All runs use the M-size model with RevIN on and a fixed training budget of 18,000 steps.

**Table 4.** N-scaling results (Transformer-M, RevIN ON, aug ON, s = 18,000 steps, seed = 42). Exception: n = 5 reports five-seed mean ± std (13.28 ± 0.12%). The n = 50 entry reports the seed-42 value (12.93%), corresponding to the best seed in Table 3; the five-seed mean for n = 50 is 13.11 ± 0.17%. Fig. 3 plots all 17 points including intermediate values not listed here.

| n | Total Buildings | NRMSE (%) | vs SOTA |
|:-:|:---------------:|:----------:|:-------:|
| 1 | 14 | 14.72 | +1.45 |
| 3 | 42 | 13.47 | +0.20 |
| 5 | 70 | 13.28 ± 0.12 | +0.01 |
| 10 | 140 | 13.18 | -0.09 |
| 20 | 280 | 13.23 | -0.04 |
| 50 | 700 | 12.93 | -0.34 |
| 70 | 980 | 13.20 | -0.07 |
| 80 | 1,120 | 13.15 | -0.12 |

Performance improves sharply from n = 1 to n = 5 (14.72% seed-42 single run to 13.28% five-seed mean). By n = 3 (42 buildings), performance reaches 13.47%, within 0.20 pp of the SOTA. At n = 5 (70 buildings), the five-seed mean of 13.28 ± 0.12% matches the SOTA (13.27%) within 0.01 pp. Beyond n = 5, gains become incremental on the single-seed curve—additional buildings yield fluctuations within ±0.15 pp rather than systematic improvement.

The n = 5 multi-seed result (13.28 ± 0.12%) confirms that the 70-building result is not a single-seed artifact. The n = 50 five-seed mean (13.11 ± 0.17%) further improves over n = 5 (13.28 ± 0.12%) by 0.17 pp, indicating that gains remain detectable through n = 50 when averaged across multiple seeds; however, the single-seed curve shows high variance in this range (±0.15 pp), so individual runs may not consistently reflect this improvement. The scaling behavior is qualitatively different from the monotonic improvement typically assumed in large-scale pretraining. The observed curve suggests diminishing returns under the current LHS design and fixed training budget.

The BB scaling experiment (BB-700 vs. BB-7K, Appendix B) shows the same saturation from the other direction: increasing from 700 to 7,000 randomly sampled BuildingsBench buildings improves performance but remains well above Korean-700 (BB-700 no-aug: 15.28% RevIN ON / 16.44% OFF; BB-7K no-aug: 14.50% RevIN ON / 15.41% OFF). Even with augmentation, BB-700 reaches only 14.26%—1.15 pp above Korean-700—indicating that additional stock-model samples alone do not close the gap.

### 4.4 Ablation Studies

Table 5 reports the two ablations conducted under the unified evaluation protocol (Section 3.4).

**Table 5.** Ablation results (val_best checkpoint, 955 buildings). Baseline: seed = 42 (12.93%). RevIN OFF reports three-seed mean, seeds 42–44 (Appendix A.2); the +1.79 pp delta therefore mixes a single-seed baseline with a three-seed mean (five-seed comparison: +1.61 pp, Section 4.2). Appendix C reports four additional ablations from an earlier evaluation pipeline.

| Experiment | NRMSE (%) | Delta | Interpretation |
|------------|:----------:|:-----:|----------------|
| Korean-700 RevIN ON (baseline) | 12.93 | — | Best configuration (seed 42, lat/lon = zero) |
| RevIN OFF (3-seed mean) | 14.72 | +1.79 | RevIN contributes 1.79 pp (1.61 pp over five-seed means) |
| Actual lat/lon coordinates | 12.93 | 0.00 | Geographic features provide no benefit with RevIN |

All four additional ablations (Appendix C) degrade performance by more than 2 pp, consistent with the broader observation that excessive exposure to synthetic patterns can harm OOD transfer: using BB-fitted Box-Cox parameters maps Korean loads outside the trained range (+3.31 pp); extending training to 168K steps causes the model to memorize synthetic artifacts (+3.09 pp); scaling to 70,000 buildings with a fixed training budget leads to under-sampling (+2.42 pp); and seasonal decomposition before RevIN disrupts OOD transfer (+3.72 pp).

---

## 5. Discussion

### 5.1 RevIN's Asymmetric Effect: Why It Helps Small Data but Hurts Large Data

Fig. 4 illustrates the asymmetry in RevIN's effect across dataset scales. On our 700-building Korean dataset, RevIN reduces NRMSE by approximately 1.6 pp in the available multi-seed comparison (14.72% three-seed mean to 13.11% five-seed mean). On the 700-building BB subset, the reduction is 1.16 pp (16.44% to 15.28%; seed 42, single run). But on the full 900K BB corpus, RevIN increases NRMSE by 0.62 pp (13.27% to 13.89%). The same technique that is beneficial at small scale is detrimental at large scale.

RevIN normalizes each 168-hour context window to zero mean and unit variance, stripping absolute load magnitude and within-window variance. The model then operates in a scale-free space, learning only temporal shape. With 700 LHS-designed buildings, the model has not seen enough magnitude variation to internalize it; RevIN solves this analytically. With 900,000 stock-model buildings each contributing roughly 2–3 sampled training windows on average, the model has already learned to exploit magnitude as a signal—a building consuming 500 kW at midnight behaves differently from one consuming 5 kW. Removing this information via RevIN discards something useful, producing a net loss.

A structural factor reinforces this mechanism. In Buildings-900K, a building's magnitude is informative about its type because HVAC, envelope, and schedule are jointly determined by stock-model parameters—magnitude and temporal shape are correlated. In our LHS design, magnitude reflects only the scale_mult parameter, which is sampled independently of the schedule parameters that determine temporal shape. RevIN's removal of magnitude therefore discards less useful information in LHS data than in stock-model data.

RevIN is therefore not universally beneficial: its value depends on whether the training data already covers the magnitude range of the evaluation set.

### 5.2 Why Operational Diversity May Matter More Than Scale

Seven hundred buildings with LHS-designed operational schedules match and in most seeds surpass 900,000 stock-model buildings on the BuildingsBench evaluation protocol (with RevIN and augmentation). Buildings-900K draws from the NREL End-Use Load Profiles for the U.S. building stock. This is a carefully constructed stock model, but a stock model nonetheless. It generates buildings whose operational parameters reflect the statistical distribution of real U.S. commercial buildings: most offices operate 08:00 to 18:00 on weekdays; most retail stores have moderate baseloads; most hospitals run continuously. Adding more buildings from this distribution increases the sample size but does not proportionally increase the diversity of temporal patterns the model encounters. Many of the 900,000 buildings share similar operational profiles because they are drawn from the same assumptions about how buildings operate.

Our 12D LHS design takes a different approach. By sampling operational parameters uniformly across their feasible ranges, we produce buildings that occupy regions of operational space rarely seen in a stock-model sample: offices that run 24 hours with 95% baseload, schools with random weekly schedule disruptions, warehouses with strong seasonal oscillation. These extreme and unusual patterns may be rare in reality, but they teach the model about the full range of temporal dynamics that building loads can exhibit. The BuildingsBench evaluation set contains real buildings, and real buildings—especially non-standard ones like data centers, 24-hour retail, or buildings with unusual HVAC systems—may have operational patterns closer to our LHS extremes than to the stock-model center.

From a scaling-law perspective, this is consistent with Shi et al. [5], who showed that scaling in time series does not follow language-model power laws. In language modeling, additional tokens more directly expand the combinatorial coverage of linguistic contexts; in building load time series, each additional building from the same stock model may add only a variation on patterns already in the training set. The marginal information content decreases rapidly.

The n-scaling analysis (Table 4, Section 4.3) provides direct evidence: SOTA-level performance emerges at just 70 buildings (n = 5 per archetype), with diminishing returns on the single-seed curve beyond 140 buildings (n = 10). Once the LHS parameter space is adequately covered, additional samples add little new temporal pattern information. This saturation differs from the monotonic gains often associated with large-scale pretraining and is consistent with learning that is governed by the number of distinct temporal patterns, not the number of examples.

### 5.3 Practical Implications

Deployment follows a straightforward path. An organization can generate several hundred parametric EnergyPlus simulations conforming to local building codes and climate, train a standard Transformer with RevIN and augmentation, and deploy zero-shot forecasting—without assembling a 900,000-building dataset or collecting extensive real-world measurements.

The computational cost difference is substantial. Our 700 EnergyPlus simulations complete in approximately 4 hours on a single workstation. Training requires 18,000 gradient steps on a single GPU (approximately 2 hours on an RTX 4090). The entire pipeline from simulation to deployed model takes less than a day. BuildingsBench's 900,000 simulations required the precomputed NREL EULP database and training at a comparable cost per step but with a dataset 1,286 times larger.

The finding that geographic features are unnecessary further simplifies deployment. Building operators need only provide recent hourly electricity measurements and timestamps. No location metadata, building-type classification, climate zone, or HVAC system information is required. RevIN helps absorb instance-level scale differences, reducing the need for explicit geographic metadata.

The climate mismatch is itself informative: both corpora train on single-country synthetic data (U.S. vs. South Korea) yet are evaluated on real buildings across four U.S. climate zones and Portugal. The U.S.-TMY/LHS ablation (Section 4.2) shows that LHS-designed schedules retain most of their advantage even under swapped weather: the five-seed mean of 13.64 ± 0.65% outperforms the matched BB-700 baseline (14.26%) by 0.62 pp, though it falls 0.53 pp short of Korean-700 (13.11 ± 0.17%) with notably wider seed variance. This pattern suggests that operational diversity is the larger contributor, but weather choice still affects both average performance and run-to-run stability.

### 5.4 Limitations

Our parametric simulations differ from BuildingsBench in schedule design (12D LHS vs. stock model), climate (Korean vs. U.S. weather), building codes, and envelope parameters. At equal scale with matched augmentation, Korean data outperforms BB data by 1.15 pp (13.11% vs. 14.26%), establishing that the data source matters. The U.S.-TMY/LHS ablation (Section 4.2) substantially reduces but does not eliminate the climate-origin confound: retraining on the same 700 LHS schedules with U.S. TMY weather yields a five-seed mean of 13.64 ± 0.65%, still outperforming BB-700 (14.26%) but falling 0.53 pp short of Korean-700 (13.11 ± 0.17%) with wider seed variance. This indicates that schedule diversity is a major contributor, but weather choice retains a measurable effect on both average performance and stability. The aug-matched BB-700 RevIN OFF experiment was not conducted; the estimated augmentation-corrected gap of approximately 0.70 pp reported in Section 4.2 assumes that augmentation provides a similar benefit regardless of the RevIN condition, an assumption that has not been empirically verified.

Controlled experiments address the two most likely alternative explanations. Applying RevIN to BB 900K degrades performance by 0.62 pp (13.27% to 13.89%), ruling out RevIN as the sole driver. Augmentation contributes 1.02 pp to BB-700 (15.28% to 14.26%) and 0.56 pp to Korean-700 (13.67% to 13.11%), yet the data-source gap remains at 1.15 pp with matched augmentation and widens to 1.61 pp without it. The NCRPS comparison (Korean-700: 7.14% vs. BB-700: 7.80%) reinforces the same ordering.

Our five-seed mean of 13.11 ± 0.17% is 0.16 pp better than the BB SOTA (13.27%), a margin comparable to the inter-seed standard deviation (0.17%). BuildingsBench does not report confidence intervals, so a formal significance test is not possible; the numerical advantage is real but modest, and the stronger claim rests on the best-seed result (12.93%, 0.34 pp better). N-scaling intermediate points (n = 2–4, 6–9, 20–40, 60–80) use seed = 42; the non-monotonic variation within ±0.15 pp at larger n likely reflects seed variance rather than a systematic trend.

All experiments use the Transformer-M architecture (15.8M parameters); generalization to PatchTST, MOIRAI, or Chronos has not been tested. On the residential segment (953 buildings), our model yields 77.71% NRMSE, comparable to the Persistence Ensemble (77.88%), confirming that zero-shot residential forecasting remains an open problem. Real-world validation beyond the BuildingsBench evaluation set (which itself comprises real metered buildings) has not been conducted; broader testing across diverse Korean building types is needed. Checkpoints are selected by validation loss on held-out Korean simulation data; no BuildingsBench test-set information influences model selection.

---

## List of Figures

**Fig. 1.** End-to-end pipeline: from building archetype selection and 12D LHS parameter sampling through EnergyPlus simulation, Box-Cox normalization, RevIN-equipped Transformer training, to zero-shot inference without geographic information.

**Fig. 2.** Zero-shot load forecasting performance (NRMSE, %) across model configurations on the 955-building evaluation set (U.S. commercial buildings and Portuguese electricity consumers). The dashed line indicates the BB SOTA-M baseline (13.27%). Korean-700 (RevIN ON, aug) achieves a five-seed mean of 13.11% (0.16 pp below the baseline, within the inter-seed standard deviation of 0.17%) and a best-seed result of 12.93% (0.34 pp below).

**Fig. 3.** N-scaling curve showing NRMSE as a function of the number of training buildings. Performance matches the SOTA from 70 buildings (n = 5), with gains becoming incremental beyond 140 buildings (n = 10). The BB SOTA-M (13.27%) baseline is shown for reference.

**Fig. 4.** RevIN's asymmetric effect across dataset scales. Green arrows indicate RevIN improvement (lower NRMSE); the red arrow indicates RevIN degradation on the full 900K BuildingsBench corpus.

---

## 6. Conclusion

Data design can substitute for scale in zero-shot building energy forecasting. Seven hundred EnergyPlus simulations, parameterized through 12-dimensional LHS to span the space of plausible commercial building operations, match and in four of five seeds surpass the 900,000-building BuildingsBench baseline. Controlled experiments at equal scale attribute the advantage primarily to data design (1.15–1.61 pp across augmentation conditions) rather than to RevIN or augmentation alone.

The n-scaling saturation near 70–140 buildings challenges the assumption that building energy foundation models require massive training corpora: once the LHS parameter space is adequately sampled, additional buildings provide diminishing new temporal pattern information rather than the monotonic gains predicted by power-law scaling. RevIN's asymmetric effect—improving small LHS-designed datasets by approximately 1.6 pp in the available multi-seed comparison while degrading the 900K BuildingsBench corpus by 0.62 pp—reveals that the value of instance normalization depends on whether the training data already spans the magnitude range of the evaluation set, not simply on dataset size.

For organizations deploying zero-shot forecasting in data-sparse regions, parametric EnergyPlus simulations designed for operational diversity represent a practical and computationally inexpensive path: a single workstation day of simulation and a few GPU-hours of training, with no location metadata or building-type classification required. A five-seed U.S.-TMY/LHS ablation supports the conclusion that the improvement is not solely driven by Korean weather: U.S.-TMY-700 (13.64 ± 0.65%) still outperforms BB-700 (14.26%), though the 0.53 pp gap relative to Korean-700 and wider seed variance indicate residual weather sensitivity. The principal open questions are the limitation to commercial buildings and the augmentation asymmetry between our pipeline and BuildingsBench. These limit the generalizability of the conclusions and call for follow-up work before the approach can be recommended across all building types.

---

## Acknowledgements

This work was supported by the Korea Institute of Energy Technology Evaluation and Planning (KETEP) grant funded by the Ministry of Trade, Industry and Energy, Republic of Korea (Grant No. RS-00238487).

## CRediT Author Statement

**Jeong-Uk Kim**: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review & editing, Visualization.

## Data Availability

The simulation pipeline, model source code, and training configurations are publicly available at https://github.com/jukkim/korean-buildingsbench. The 700-building parametric simulation dataset can be regenerated from the provided pipeline (EnergyPlus IDF generation, simulation, and post-processing scripts in Section 3.1) or is available from the corresponding author upon request. Pretrained checkpoints for the primary results (Korean-700 seeds 42--46, BB-700, BB 900K + RevIN, US-TMY-700 seeds 42--46) and n-scaling intermediate points (n = 2--4, 6--9, 20, 30, 40, 60, 70, 80) are available from the corresponding author upon request. The BuildingsBench evaluation data can be downloaded from the NREL Open Energy Data Initiative (https://data.openei.org/submissions/5859) under a CC-BY 4.0 license.

## Declaration of Competing Interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

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

§ Two-seed mean (seeds 42, 44); seed 43 NCRPS not evaluated due to broken original checkpoint.

### A.3 Korean-700 RevIN ON (s = 16,000, 5 seeds)

Five-seed evaluation at s = 16,000 steps yields 13.13 ± 0.15%, confirming robustness to modest variation in training duration. The s = 18,000 setting (13.11 ± 0.17%) is used as the primary result throughout the paper.

| Seed | NRMSE (%) |
|:----:|:----------:|
| 42 | 12.89 |
| 43 | 13.24 |
| 44 | 13.16 |
| 45 | 13.26 |
| 46 | 13.12 |
| **Mean ± Std** | **13.13 ± 0.15** |

### A.4 BB 900K + RevIN

The BuildingsBench 900K model was retrained from scratch with RevIN enabled (identical architecture, optimizer, and training schedule to the original). The resulting NRMSE of 13.89% on the 955-building evaluation set represents a 0.62 pp degradation relative to the original 13.27%. This single-run result uses the same seed and hyperparameters as the original BuildingsBench training.

---

## Appendix B: BB-700 and BB-7K Scaling

| Configuration | N Buildings | RevIN | Aug | NRMSE (%) | NCRPS (%) |
|--------------|:-----------:|:-----:|:--:|:----------:|:---------:|
| BB-700 | 700 | ON | OFF | 15.28 | — |
| BB-700 (aug-matched) | 700 | ON | ON | 14.26 | 7.80 |
| BB-700 | 700 | OFF | OFF | 16.44 | — |
| BB-7K | 7,000 | ON | OFF | 14.50 | — |
| BB-7K | 7,000 | OFF | OFF | 15.41 | — |

A 10x increase in BuildingsBench buildings (700 to 7,000) reduces NRMSE by 0.78 pp with RevIN, no aug (15.28% to 14.50%). BB-700 with augmentation (14.26%) narrowly outperforms BB-7K without augmentation (14.50%), though this comparison conflates augmentation and scale effects. Regardless, BB-700+aug (14.26%) remains 1.15 pp above Korean-700 (13.11%), and even BB-7K (14.50%) does not approach the performance of 700 LHS-designed buildings with augmentation.

All results verified through unified evaluation (torch 2.0.1, autocast, identical protocol as BB SOTA reproduction).

---

## Appendix C: Additional Ablation Results (Earlier Pipeline)

The four experiments below were conducted with an earlier evaluation pipeline prior to the adoption of the val_best checkpoint and unified 955-building evaluation. The directions of degradation are reliable, but exact NRMSE values carry approximately ±0.3 pp additional uncertainty relative to Table 3 and Table 5.

| Experiment | NRMSE (%) | Delta vs Table 3 baseline | Note |
|------------|:----------:|:-------------------------:|------|
| BB Box-Cox (BB-fitted scale/location) | 16.24 | +3.31 | Korean loads mapped outside trained range |
| 4× training tokens (168K steps, ~84 epochs) | 16.02 | +3.09 | Overfitting to synthetic artifacts |
| 5K cap per archetype (70K buildings total) | 15.35 | +2.42 | Fixed-budget under-sampling at scale |
| Seasonal decomposition + RevIN | 16.65 | +3.72 | Trend-seasonal separation hurts OOD transfer |

Each failure mode reflects a distinct mechanism. (1) Replacing the Korean-fitted Box-Cox parameters with the BB-fitted values maps Korean load magnitudes into a different region of the transformed space, causing a systematic mismatch between the training distribution and the expected input range (+3.31 pp). (2) Extending training to 168K steps (~84 epochs over 700 buildings) allows the model to memorize simulation-specific schedule artifacts—ramp shapes, exact transition widths, and load plateaus—that do not generalize to real buildings (+3.09 pp). (3) Scaling to 70,000 buildings with the same 18,000-step budget reduces each building's contribution to roughly 0.03 sampled training windows on average, insufficient to learn building-specific patterns while still constrained by the same total optimization budget (+2.42 pp). (4) Applying seasonal decomposition before RevIN separates trend and seasonal components, disrupting the temporal shape diversity that LHS is designed to provide; when RevIN subsequently normalizes the residual, the model loses access to the periodicity signals that aid OOD generalization (+3.72 pp).
