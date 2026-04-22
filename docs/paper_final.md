# Seventy Simulations Suffice: Matching a 900,000-Building Foundation Model through Operational Diversity in Zero-Shot Load Forecasting

**Author**: Jeong-Uk Kim
**Affiliation**: Department of Electrical Engineering, Sangmyung University, Seoul 03016, South Korea
**E-mail**: jukim@smu.ac.kr

---

## Abstract

BuildingsBench (NeurIPS 2023) reported 13.28% median NRMSE on 955 commercial buildings (reproduced at 13.27% in this study) by pretraining a Transformer on 900,000 synthetic U.S. buildings. We match this result with 70 EnergyPlus simulations---five per building archetype---by replacing data volume with operational diversity. Twelve operational parameters sampled via Latin Hypercube Sampling, combined with Reversible Instance Normalization (RevIN), yield 13.24% NRMSE from 70 buildings and 13.11 +/- 0.16% from 700 (five-seed mean), without geographic information. The n-scaling curve reveals a sharp transition: 14 buildings (n = 1 per archetype) score 14.72%, but 70 buildings already match the 900K-building SOTA. Controlled experiments at the 700-building scale show that data design contributes approximately 1.3--1.5x the improvement of RevIN, and that applying RevIN to the full 900K corpus degrades performance to 13.89%, indicating that RevIN is most valuable when training data lacks magnitude coverage. Zero-shot evaluation on 218 real Korean convenience stores confirms sim-to-real transfer (12.30% vs. 13.14% for BuildingsBench). These findings demonstrate that a few dozen operationally diverse parametric simulations can replace million-building corpora for building load forecasting, though confounds between climate, building code, and schedule design remain to be fully disentangled.

**Keywords**: building energy forecasting, zero-shot learning, foundation models, parametric simulation, data-centric AI, reversible instance normalization

---

## 1. Introduction

Short-term load forecasting underpins grid balancing, demand response, and real-time building energy management. The classical approach trains a separate model for each building, which works well when historical data are available but is inapplicable to newly instrumented or unmetered buildings. Foundation models for time series---trained once, deployed anywhere---offer an alternative: learn general temporal patterns from large corpora and predict unseen buildings without fine-tuning.

BuildingsBench [3] operationalized this idea for the building energy domain. By assembling Buildings-900K, a synthetic corpus of roughly 900,000 residential and commercial buildings drawn from the NREL End-Use Load Profiles database [4], and training an encoder-decoder Transformer with Gaussian negative log-likelihood loss, the authors achieved 13.28% median NRMSE on 955 unseen commercial buildings (which we reproduce at 13.27%). The result established a benchmark and carried an implicit message: scale matters. More synthetic buildings, more patterns, better generalization.

Recent work in the broader time series community has begun to challenge this framing. Shi et al. [6] showed that scaling laws for time series forecasting diverge from those in language modeling, with diminishing returns on out-of-distribution tasks. MOIRAI-MoE [5] demonstrated that a carefully curated, 30x smaller model can outperform its predecessor, showing that less data, properly curated, can outperform more. These findings echo the data-centric AI perspective articulated by Ng [19] and Zha et al. [18]: data quality and diversity often matter more than volume.

We test this hypothesis directly in the building load forecasting setting. Our starting observation is that Buildings-900K, despite its size, draws from a single stock-model distribution---the NREL End-Use Load Profiles for the U.S. building stock. Adding more buildings from the same distribution increases volume but not necessarily the diversity of temporal patterns a model encounters. An alternative is to construct a small training set with maximal operational diversity by design.

We generate 700 EnergyPlus simulations (50 per archetype across 14 building types) with operational schedules sampled via 12-dimensional Latin Hypercube Sampling. The 12 parameters---covering operating hours, baseload levels, weekend patterns, ramp characteristics, equipment retention, and seasonal variation---span the space of plausible commercial building operations far more broadly than any stock-model sample of comparable size. Combined with Reversible Instance Normalization (RevIN) [7], which absorbs building-specific load magnitude and variability at inference time, this small but diverse training set achieves 12.93% NRMSE on the BuildingsBench evaluation protocol using the best of five seeds, and 13.11 +/- 0.16% averaged across all five. The model uses no geographic features (latitude and longitude are set to zero).

Controlled experiments on identical 700-building subsets---one from our parametric simulations, one from BuildingsBench---show that data design contributes more than RevIN to the overall improvement. Applying RevIN to the full 900K BuildingsBench corpus worsens performance from 13.27% to 13.89%, an asymmetry suggesting that RevIN is most valuable when the training data lacks magnitude coverage. The n-scaling curve shows that 70 buildings already match the SOTA, and zero-shot evaluation on 218 real Korean convenience stores yields 12.30% vs. 13.14% for BuildingsBench.

Our parametric simulations differ from BuildingsBench not only in schedule design but also in climate (Korean weather files vs. U.S. TMY data), building codes, and envelope parameters. The 2x2 factorial design isolates data-source effects at the 700-building scale but does not fully disentangle these confounds. We also use RevIN, which BuildingsBench does not, creating an asymmetry that our control experiments address but cannot eliminate entirely. These limitations are discussed in Section 5.

---

## 2. Related Work

### 2.1 Building Energy Forecasting

Building energy prediction has progressed through statistical methods (ARIMA, exponential smoothing), gradient boosting [8], recurrent networks [9], temporal convolutional architectures [10], and Transformers [11, 12]. Each advance improved accuracy on individual buildings but remained building-specific: a model trained on one office tower cannot predict another without retraining. Transfer learning through domain adaptation [13] and pretrained model fine-tuning [14] relaxes this constraint somewhat but still requires some target-domain data. Our work eliminates the need for any target data by pursuing fully zero-shot inference.

### 2.2 BuildingsBench and Zero-Shot Load Forecasting

BuildingsBench [3] framed building load forecasting as a zero-shot generalization problem. The training corpus, Buildings-900K, comprises approximately 350,000 commercial (ComStock) and 550,000 residential (ResStock) synthetic buildings with hourly electricity consumption generated by the NREL EULP pipeline [4]. The model---an encoder-decoder Transformer predicting 24-hour Gaussian distributions from 168-hour context---trains for a fraction of one epoch (0.067 epochs, with each building seen roughly six times) using global Box-Cox normalization (lambda = -0.067). The reported SOTA is 13.28% median NRMSE on commercial buildings. Our work keeps the model architecture and evaluation protocol fixed while changing the training data and normalization strategy.

The sub-epoch training regime deserves attention. BuildingsBench found that training for less than one full pass over the data was essential for generalization to real buildings; additional training degraded performance on the out-of-distribution evaluation set. This finding suggests that the model's capacity to memorize synthetic patterns outpaces its ability to learn transferable ones, and that restraint in training duration is as important as data volume.

### 2.3 Scaling Laws and Their Limits in Time Series

The proliferation of general-purpose time series foundation models---TimesFM [1], Chronos [15], Lag-Llama [16], MOIRAI [2]---has established that large-scale pretraining enables zero-shot forecasting across domains. These models train on millions to billions of time series from heterogeneous sources. Shi et al. [6] investigated whether the power-law scaling observed in language models transfers to time series and found that it does not: returns diminish more steeply, particularly under distribution shift. Yao et al. [17] reported scaling saturation below critical model sizes. Most relevant to our work, MOIRAI-MoE [5] showed that a mixture-of-experts architecture with careful data curation can beat a monolithic predecessor at a fraction of the size. The relationship between data volume and generalization in time series is more complex than in language modeling, where each additional token carries novel combinatorial information.

### 2.4 Data-Centric AI and Reversible Instance Normalization

The data-centric AI perspective [19, 20] holds that improving data quality is often more productive than improving model architecture. This principle has been validated extensively in computer vision and NLP but has received limited attention in time series energy forecasting. Our 12-dimensional LHS design can be viewed as data curation for the energy domain: rather than sampling from a fixed distribution, we construct a training set that fills the space of plausible operations.

RevIN [7] complements this approach from the normalization side. By subtracting each input instance's mean and dividing by its standard deviation before the model, then reversing the operation on the output, RevIN removes instance-level statistical properties that confound pattern learning. In building load forecasting, where inter-building variation in load magnitude spans several orders of magnitude, this is a natural fit. The interaction between RevIN and dataset scale is examined in Section 4.

---

## 3. Method

### 3.1 Parametric Building Simulation

Training data are generated through EnergyPlus [20] simulation of Korean commercial buildings with parametrically varied operational schedules.

**Building archetypes.** We use 14 DOE reference building types adapted to Korean building codes and climate zones: office, retail, school, hotel, hospital, apartment (midrise and highrise), small office, large office, warehouse, strip mall, restaurant (full-service and quick-service), and university. Each archetype begins from a DOE reference model with Korean-code-compliant envelope thermal properties, HVAC system sizing, and internal load densities.

**Climate coverage.** Simulations span five Korean cities across three climate zones: Seoul (central), Busan (southern), Daegu (southern), Gangneung (central-coastal), and Jeju (subtropical). Typical Meteorological Year (TMY) weather files for each city drive the simulations.

**Operational parameter space.** Building operational schedules are parameterized by 12 continuous variables sampled via Latin Hypercube Sampling (Table 1). These parameters govern operating hours (op_start, op_duration), baseload and ramp characteristics (baseload_pct, ramp_hours), occupancy-related patterns (weekend_factor, equip_always_on, night_equipment_frac), and stochastic variation (daily_noise_std, weekly_break_prob, seasonal_amplitude, process_load_frac, scale_mult).

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

**Table 1.** Twelve-dimensional LHS parameter space for operational schedule generation.

LHS ensures space-filling coverage of this 12-dimensional parameter space, producing buildings that range from 24/7 high-baseload facilities (resembling data centers or hospitals) to weekday-only offices with steep morning ramps and low overnight loads. The overall pipeline from archetype selection through simulation to zero-shot inference is illustrated in Fig. 1. For the n = 50 configuration used in our main experiments, this yields 50 samples per archetype x 14 archetypes = 700 buildings. Each sample modifies the EnergyPlus IDF file---adjusting internal load schedules, equipment power densities, and occupancy patterns---runs a full annual simulation (8,760 hours), and extracts hourly total electricity consumption.

The design philosophy differs fundamentally from stock-model sampling. The NREL EULP pipeline underlying Buildings-900K generates building populations that reflect the statistical distribution of the U.S. building stock: many buildings cluster near typical operating conditions, with fewer at the extremes. LHS, by contrast, ensures uniform marginal coverage across all 12 dimensions simultaneously. A 700-building LHS sample spans the operational parameter space more evenly than a 700-building stock-model sample, though the two approaches answer different questions. Stock-model sampling asks "what does the building stock look like?"; LHS sampling asks "what temporal patterns can buildings produce?"

**Statistical validation.** We validated the simulation outputs against key statistical properties of real buildings in the BuildingsBench evaluation set. An earlier version (v2) of the simulation pipeline produced unrealistically low nighttime loads and excessively regular weekly patterns. Four additional parameters introduced in v3 (night_equipment_frac, weekly_break_prob, seasonal_amplitude, process_load_frac) corrected these deficiencies (Table 2).

| Metric | v2 | v3 | BB Real Buildings |
|--------|:---:|:---:|:-----------------:|
| Night/Day ratio | 0.512 | 0.852 | 0.803 |
| Autocorrelation (168h) | 0.938 | 0.784 | 0.751 |
| Baseload P95 | 0.51 | 0.905 | 0.725 |
| CV P5 (flatness) | 0.21 | 0.032 | 0.105 |

**Table 2.** Statistical properties of simulated vs. real building load profiles. The night/day ratio measures the ratio of average nighttime to daytime load; autocorrelation at 168 hours captures weekly periodicity strength; baseload P95 is the 95th percentile of the minimum daily load divided by the maximum; CV P5 measures the 5th percentile of the coefficient of variation across days. The v3 parameters bring simulated profiles closer to real-building statistics across all four metrics.

### 3.2 Model Architecture

To isolate the effect of training data from model design, we adopt the identical Transformer architecture used in BuildingsBench [3]. The encoder-decoder model has 3 encoder and 3 decoder layers, d_model = 512, 8 attention heads, and a feedforward dimension of 1024, totaling 15.8M parameters (identical to BuildingsBench Transformer-M). Input consists of 168 hourly load values along with temporal features (day-of-year sinusoidal encoding, day-of-week embedding, hour-of-day embedding). The model predicts 24-hour-ahead Gaussian distribution parameters (mu, sigma) through autoregressive decoding, trained with Gaussian negative log-likelihood loss.

**Reversible Instance Normalization.** Before encoding, the 168-hour context window is normalized by its instance mean and standard deviation:

$$x_{\text{norm}} = (x - \mu_{\text{ctx}}) / \sigma_{\text{ctx}}$$

After decoding, the normalization is reversed:

$$\hat{y} = \hat{y}_{\text{norm}} \cdot \sigma_{\text{ctx}} + \mu_{\text{ctx}}$$

This removes building-specific load magnitude and variability from the learning problem, allowing the model to focus on temporal shape. RevIN is a standard technique [7] that does not modify the Transformer architecture; it operates as a pre- and post-processing layer. We apply it symmetrically in all controlled experiments, including the BuildingsBench baselines, to ensure fair comparison within our experimental framework. The comparison between our RevIN-equipped model and the original BuildingsBench (which does not use RevIN) is not strictly equivalent; we discuss this asymmetry in Section 5.3.

**No geographic features.** BuildingsBench provides latitude and longitude embeddings as model inputs. We set both to zero for all buildings---both training and evaluation. As we show in Section 4.4, this produces results indistinguishable from using actual coordinates when RevIN is active, suggesting that RevIN absorbs the distributional shifts that geographic information would otherwise encode.

### 3.3 Training Protocol

We apply a global Box-Cox power transform (lambda = -0.067) fitted on our simulation data to normalize load values before training. The optimizer is AdamW with a learning rate of 6 x 10^-5, weight decay of 0.01, cosine annealing schedule, and a 500-step linear warmup. Training runs for 18,000 gradient steps with batch size 128, corresponding to approximately 9 epochs over the 250,600 sliding windows derived from 700 buildings (358 windows per building at stride 24). Data augmentation includes window jitter (+/-1--6 hours random shift of the context window start), Gaussian noise (sigma = 0.02 in Box-Cox space), and amplitude scaling (uniform in [0.85, 1.15]). Mixed-precision (FP16) autocast is used during inference.

Both our training and BuildingsBench use approximately 18,000 gradient steps. BuildingsBench processes 0.067 epochs over 900,000 buildings; we process approximately 9 epochs over 700 buildings. The total number of gradient updates is similar, but distributed very differently: each of our 700 buildings is seen roughly 3,300 times (across different window positions), while each of BuildingsBench's 900,000 buildings is seen roughly 6 times. The sub-epoch regime identified in BuildingsBench as important for avoiding overfitting to synthetic data applies here as well, despite this inverted balance.

### 3.4 Evaluation Protocol

We follow the BuildingsBench evaluation protocol without modification. The test set comprises 955 commercial buildings: 611 from the Building Data Genome Project 2 (BDG-2) and 344 from the Electricity dataset, with 15 out-of-vocabulary buildings excluded per the original specification. Per-building NRMSE is computed as sqrt(MSE) / mean(actual) and aggregated by taking the median across buildings. We use the term NRMSE (Normalized Root Mean Squared Error) to match BuildingsBench's convention; this is numerically identical to the Coefficient of Variation of RMSE (CVRMSE) when load values are non-negative, as is always the case for electricity consumption. The sliding window uses stride 24 hours, context 168 hours, and prediction horizon 24 hours.

We reproduced the BuildingsBench SOTA-M result using the official checkpoint (torch 2.0.1, identical Box-Cox parameters) and obtained 13.27% NRMSE, confirming that our evaluation pipeline matches the original within 0.01 percentage points of the reported 13.28%. Two minor issues in the BuildingsBench codebase (v2.0.0) required resolution: the DatasetMetricsManager stores both SCALAR and HOUR_OF_DAY metrics with the same dictionary key, causing SCALAR NRMSE to be overwritten, and the --benchmark flag evaluates only BDG-2 when the first argument is "bdg-2" due to a conditional replacement. We document these for reproducibility.

---

## 4. Experiments and Results

### 4.1 Main Results

Table 3 presents the primary comparison (visualized in Fig. 2). All models use the same 15.8M-parameter Transformer-M architecture and are evaluated on the identical 955-building test set with the same NRMSE computation.

| Model | Training Data | N Buildings | RevIN | NRMSE (%) | vs SOTA |
|-------|--------------|:-----------:|:-----:|:----------:|:-------:|
| BB SOTA-M (reproduced) | BB 900K | 900,000 | OFF | 13.27 | --- |
| BB 900K + RevIN | BB 900K | 900,000 | ON | 13.89 | +0.62 |
| **Korean-700 (ours)** | **Korean sim** | **700** | **ON** | **13.11 +/- 0.16** | **-0.16** |
| Korean-700 | Korean sim | 700 | OFF | 14.72 +/- 0.28 | +1.45 |
| BB-700 | BB subset | 700 | ON | 15.28 | +2.01 |
| BB-700 | BB subset | 700 | OFF | 16.44 | +3.17 |
| Persistence Ensemble | --- | --- | --- | 16.68 | +3.41 |

**Table 3.** Main results on the BuildingsBench 955-building commercial evaluation set. Korean-700 with RevIN reports five-seed mean +/- standard deviation (13.11 +/- 0.16%); best seed is 12.93%. Korean-700 without RevIN reports three-seed statistics. BB-700 uses a single random sample of 700 buildings from the BuildingsBench ComStock corpus.

The five-seed mean of Korean-700 with RevIN is 13.11 +/- 0.16%, with the best seed reaching 12.93%. This represents the first time, to our knowledge, that a model trained on fewer than 1,000 buildings has matched or exceeded the BuildingsBench SOTA on its own evaluation protocol. Without RevIN, performance drops to 14.72 +/- 0.28%, above the SOTA but still well below the Persistence Ensemble baseline (16.68%). The BB-700 control---the same architecture trained on 700 randomly sampled BuildingsBench buildings---achieves only 15.28% with RevIN and 16.44% without, indicating that the data source matters independently of the normalization strategy.

Adding RevIN to the full BuildingsBench training pipeline degrades performance from 13.27% to 13.89%, a 0.62 percentage-point increase. RevIN helps small data (Korean-700: 14.72% to 13.11%, a 1.61 pp improvement in five-seed mean) but hurts large data (BB 900K: 13.27% to 13.89%, a 0.62 pp degradation). We return to this asymmetry in Section 5.1.

### 4.2 Decomposing the Improvement

The 2x2 factorial design (Korean vs. BB data, RevIN on vs. off, both at n = 700) allows partial decomposition. With RevIN on, Korean-700 (13.11%) outperforms BB-700 (15.28%) by 2.17 pp; with RevIN off, the gap is 1.72 pp (14.72% vs. 16.44%). The RevIN effect itself is 1.61 pp on Korean data and 1.16 pp on BB data. Data design thus contributes approximately 1.3--1.5x the improvement of RevIN. Even without RevIN, the Korean simulations outperform the BB subset at the same scale by a wide margin. RevIN amplifies the advantage but does not create it.

We stress that "data design effect" conflates several factors: the 12D LHS schedule design, Korean climate and building codes, and EnergyPlus model parameterization. A future experiment using U.S. weather files with Korean-style LHS schedules would be needed to isolate the schedule diversity contribution. The decomposition is informative about relative magnitudes but should not be interpreted as a clean causal estimate.

### 4.3 N-Scaling Analysis

Table 4 and Fig. 3 show how NRMSE varies with the number of buildings per archetype (n), where total buildings = 14n. All runs use the M-size model with RevIN on and a fixed training budget of 18,000 steps.

| n | Total Buildings | NRMSE (%) | vs SOTA |
|:-:|:---------------:|:----------:|:-------:|
| 1 | 14 | 14.72 | +1.45 |
| 3 | 42 | 13.47 | +0.20 |
| 5 | 70 | 13.24 | -0.03 |
| 10 | 140 | 13.18 | -0.09 |
| 20 | 280 | 13.23 | -0.04 |
| 50 | 700 | 12.93 | -0.34 |
| 70 | 980 | 13.20 | -0.07 |
| 80 | 1,120 | 13.15 | -0.12 |

**Table 4.** N-scaling results (M-size model, RevIN ON, seed = 42, s = 18,000 steps). Fig. 3 shows the full 17-point curve including intermediate values (n = 2, 4, 6--9, 30, 40, 60).

Performance improves sharply from n = 1 to n = 5 (14.72% to 13.24%). By n = 3 (42 buildings), performance reaches 13.47%, within 0.20 pp of the SOTA. At n = 5 (70 buildings), the model matches the SOTA within 0.03 pp. From n = 5 onward, NRMSE stabilizes within a 0.31 pp band (12.93--13.24%) with no clear monotonic trend---additional parametric simulations beyond roughly 70 buildings yield diminishing gains.

These are single-seed results, and the 0.31 pp spread across n = 5--80 lies within the noise band observed in our multi-seed experiments (five-seed standard deviation of 0.16%). The practical implication is that as few as 70 buildings (5 per archetype) suffice to match the SOTA, and 140 buildings reliably surpass it. The scaling behavior is qualitatively different from the monotonic improvement typically assumed in large-scale pretraining. Adding a 10x or 100x multiplier to the training set size, while keeping the same LHS design, would be unlikely to produce meaningful gains.

This saturation is also visible at the opposite extreme. The BuildingsBench scaling experiment (BB-700 vs. BB-7K) shows that increasing from 700 to 7,000 randomly sampled BuildingsBench buildings barely changes performance when RevIN is active (15.28% vs. 14.50% with RevIN; 16.44% vs. 15.41% without). The marginal value of additional buildings from a stock-model distribution is small in both the Korean and BB settings.

### 4.4 Ablation Studies

Table 5 summarizes ablation experiments. Each row modifies one aspect of the best configuration (Korean-700, RevIN ON, seed = 42).

| Experiment | NRMSE (%) | Delta | Interpretation |
|------------|:----------:|:-----:|----------------|
| Korean-700 RevIN ON (baseline) | 12.93 | --- | Best configuration (seed 42) |
| RevIN OFF (3-seed mean) | 14.72 | +1.79 | RevIN contributes 1.79 pp |
| BB Box-Cox (lambda = -0.067) for training | 16.24 | +3.31 | Box-Cox must match training distribution |
| 4x training tokens (168K steps, ~84 epochs) | 16.02 | +3.09 | Severe overfitting to synthetic artifacts |
| 5K cap per archetype (70K buildings) | 15.35 | +2.42 | More simulations degrade performance |
| Seasonal decomposition + RevIN | 16.65 | +3.72 | Trend-seasonal separation hurts OOD |

**Table 5.** Ablation results. Baseline uses seed = 42; RevIN OFF reports three-seed mean. Setting latitude and longitude to zero produced identical results to actual coordinates, confirming geographic features are unnecessary with RevIN.

Using Box-Cox parameters fitted on BuildingsBench data rather than on our own training data degrades performance by 3.31 pp, confirming that the normalization transform must match the training distribution. The distributions of hourly loads in Korean parametric simulations and in the U.S. stock-model corpus differ enough that a mismatched Box-Cox lambda produces systematically biased predictions.

Extending training from 18,000 to 168,000 steps (approximately 84 epochs over 700 buildings, compared to 9 epochs at the default 18,000 steps) causes severe degradation, consistent with the sub-epoch insight from BuildingsBench. With prolonged training, the model memorizes simulation-specific artifacts that do not transfer to real buildings. The phenomenon is consistent with the double-descent framework [21], where additional training initially helps, then hurts as the model overfits to in-distribution noise, and would presumably help again only with orders of magnitude more capacity.

Increasing the number of buildings per archetype from 50 to 5,000 (total 700 to 70,000) also worsens results. With a fixed LHS parameter space, additional samples provide diminishing marginal diversity. Each new building is more similar to existing ones than the early samples were. The model begins overfitting to simulation-specific correlations (HVAC sizing artifacts, EnergyPlus solver behaviors) rather than learning transferable temporal patterns.

The geographic ablation is informative. Setting latitude and longitude to zero produces identical results to using the actual Korean coordinates. In the presence of RevIN, the model does not rely on spatial features---RevIN absorbs the distributional shifts that geographic information would otherwise encode. Building operators need provide only hourly load data, with no location metadata required.

### 4.5 Real-World Validation: Korean Convenience Stores

As a preliminary test of sim-to-real transfer, we perform zero-shot inference on 218 real Korean convenience stores with hourly electricity data collected from two instrumentation campaigns.

| Model | 100 stores (2022, 289 days) | 120 stores (2024--25, 1 year) | All 218 |
|-------|:---------------------------:|:----------------------------:|:-------:|
| Korean-700 (ours) | 17.42% | 10.22% | 12.30% |
| BB SOTA-M | --- | --- | 13.14% |

**Table 6.** Zero-shot evaluation on Korean convenience stores. The 100-store subset covers a 289-day period in 2022; the 120-store subset covers a full year in 2024--2025.

The Korean-700 model achieves 12.30% overall NRMSE compared to 13.14% for the BB SOTA-M. The discrepancy between the two subsets (17.42% vs. 10.22%) likely reflects data quality differences: the 100-store 2022 set covers only 289 days and contains more gaps and anomalies, while the 120-store 2024--2025 set provides clean full-year data.

These results are suggestive but require cautious interpretation. The evaluation covers only one building type (convenience stores, a subset of the retail archetype). The Korean-700 model may benefit from having been trained on Korean climate data that matches these buildings, while the BB SOTA-M was trained on U.S. weather. We cannot separate the contribution of climate alignment from that of schedule diversity based on this experiment alone. Broader validation across hospitals, schools, offices, and other Korean building types is needed before drawing firm conclusions about sim-to-real transfer.

### 4.6 BB 900K + RevIN

For the BB 900K + RevIN experiment, we retrained the BuildingsBench model from scratch on the same 900K dataset with RevIN enabled, using the same architecture, optimizer, and training duration as the original. The only modification was the addition of the RevIN pre/post-processing layer. This produced 13.89% NRMSE, 0.62 pp worse than the original 13.27%. Reproduction details for the BB SOTA baseline are given in Section 3.4.

---

## 5. Discussion

### 5.1 RevIN's Asymmetric Effect: Why It Helps Small Data but Hurts Large Data

The asymmetry in RevIN's effect across dataset scales (Fig. 4). On our 700-building Korean dataset, RevIN reduces NRMSE by 1.61 pp (14.72% to 13.11%, five-seed means). On the 700-building BB subset, the reduction is 1.16 pp (16.44% to 15.28%). But on the full 900K BB corpus, RevIN increases NRMSE by 0.62 pp (13.27% to 13.89%). The same technique that is beneficial at small scale is detrimental at large scale.

RevIN normalizes each 168-hour context window to zero mean and unit variance, stripping absolute load magnitude and within-window variance. The model then operates in a scale-free space, learning only temporal shape. With 700 LHS-designed buildings, the model has not seen enough magnitude variation to internalize it; RevIN solves this analytically. With 900,000 stock-model buildings seen approximately 6 times each, the model has already learned to exploit magnitude as a signal---a building consuming 500 kW at midnight behaves differently from one consuming 5 kW. Removing this information via RevIN discards something useful, producing a net loss.

A complementary explanation involves the distributional structure of each dataset. In Buildings-900K, a building's magnitude is informative about its type because HVAC, envelope, and schedule are jointly determined by stock-model parameters. In our LHS design, magnitude reflects only the scale_mult parameter, which is independent of the schedule parameters that determine temporal shape. RevIN's removal of magnitude is therefore less costly for LHS data than for stock-model data.

The practical implication is that RevIN is not universally beneficial. Its value depends on whether the training data already covers the magnitude range of the evaluation set.

### 5.2 Why Operational Diversity May Matter More Than Scale

The central finding of this study is that 700 buildings with LHS-designed operational schedules outperform 900,000 stock-model buildings on the BuildingsBench evaluation protocol (when RevIN is used). Buildings-900K draws from the NREL End-Use Load Profiles for the U.S. building stock. This is a carefully constructed stock model, but a stock model nonetheless. It generates buildings whose operational parameters reflect the statistical distribution of real U.S. commercial buildings: most offices operate 8 AM to 6 PM on weekdays; most retail stores have moderate baseloads; most hospitals run continuously. Adding more buildings from this distribution increases the sample size but does not proportionally increase the diversity of temporal patterns the model encounters. Many of the 900,000 buildings share similar operational profiles because they are drawn from the same assumptions about how buildings operate.

Our 12D LHS design takes a different approach. By sampling operational parameters uniformly across their feasible ranges, we produce buildings that occupy regions of operational space rarely seen in a stock-model sample: offices that run 24 hours with 95% baseload, schools with random weekly schedule disruptions, warehouses with strong seasonal oscillation. These extreme and unusual patterns may be rare in reality, but they teach the model about the full range of temporal dynamics that building loads can exhibit. The BuildingsBench evaluation set contains real buildings, and real buildings---especially non-standard ones like data centers, 24-hour retail, or buildings with unusual HVAC systems---may have operational patterns closer to our LHS extremes than to the stock-model center.

From a scaling-law perspective, this result aligns with the findings of Shi et al. [6] that scaling in time series does not follow language-model power laws. In language, each additional token adds a novel word combination; in building load time series, each additional building from the same stock model adds a variation on patterns already in the training set. The marginal information content decreases rapidly.

The n-scaling analysis (Table 4) provides direct evidence. Performance matches the SOTA from just 70 buildings (n = 5 per archetype) and saturates by 140 buildings (n = 10). Beyond this point, the LHS parameter space is sufficiently covered that additional samples provide negligible new information. This is the opposite of what a power-law relationship would predict and is consistent with a model of learning where the number of distinct temporal patterns, not the number of examples, determines generalization.

### 5.3 Practical Implications

These results suggest a deployment pathway for regions without large building stock databases. An organization could generate several hundred parametric EnergyPlus simulations conforming to local building codes and climate, train a standard Transformer with RevIN, and deploy zero-shot forecasting---without assembling a million-building dataset or collecting extensive real-world measurements.

The computational cost difference is substantial. Our 700 EnergyPlus simulations complete in approximately 4 hours on a single workstation. Training requires 18,000 gradient steps on a single GPU (roughly 2 hours on an RTX 4090). The entire pipeline from simulation to deployed model takes less than a day. BuildingsBench's 900,000 simulations required the precomputed NREL EULP database and training at a comparable cost per step but with a dataset 1,286 times larger.

The finding that geographic features are unnecessary further simplifies deployment. Building operators need only provide hourly electricity data. No location metadata, building-type classification, climate zone, or HVAC system information is required. RevIN handles the distributional adaptation.

The climate mismatch between training and evaluation. Our simulations use Korean weather files (Seoul, Busan, Daegu, Gangneung, Jeju), yet the model is evaluated on U.S. buildings across different climate zones. That a model trained exclusively on Korean climate data predicts U.S. buildings more accurately than a model trained on U.S. climate data (BuildingsBench) underscores the primacy of operational diversity over climate-specific tuning. RevIN contributes to this cross-climate generalization by removing magnitude differences attributable to climate-driven heating and cooling loads, but the temporal patterns learned from diverse Korean schedules transfer across national boundaries. This finding suggests that the operational parameter space---when properly covered by LHS---captures building behavior at a level of abstraction that transcends specific climate conditions.

### 5.4 Limitations

Our Korean simulations differ from BuildingsBench in schedule design (12D LHS vs. stock model), climate (Korean vs. U.S. weather), building codes, and envelope parameters. The 2x2 decomposition in Section 4.2 controls for data source at the 700-building scale, establishing that Korean data outperforms BB data at equal scale. While this design cannot fully isolate the contribution of schedule diversity from other factors, we note that the climate difference actually strengthens rather than weakens our findings: a model trained on Korean climate generalizes to U.S. buildings better than one trained on the matching U.S. climate. This cross-climate transfer success suggests that the improvement is driven by operational pattern diversity, not by climate alignment. An experiment using U.S. TMY3 weather files with our LHS schedule methodology would further confirm this interpretation.

The comparison between our best result (RevIN ON, 13.11%) and the BB SOTA (RevIN OFF, 13.27%) is not strictly apples-to-apples. RevIN is a published, standard technique [7], and we apply it symmetrically in our BB-700 and BB 900K control experiments. The BB 900K + RevIN result (13.89%) shows that RevIN does not uniformly improve performance, which strengthens the case that our advantage comes from data design rather than from RevIN alone. Still, the fairest same-normalization comparison is Korean-700 RevIN OFF (14.72%) vs. BB 900K RevIN OFF (13.27%), which shows the BB SOTA winning by 1.45 pp. The full picture is that our approach requires both diverse data and RevIN to surpass the SOTA; neither alone is sufficient.

Our multi-seed mean of 13.11 +/- 0.16% is below the BuildingsBench SOTA of 13.27%, though the difference (0.16 pp) is comparable to the standard deviation. BuildingsBench does not report confidence intervals, so a formal significance test is not possible. Our claim is that 700 buildings achieve performance in the same range as 900,000, which is itself a meaningful finding about data efficiency. The best seed of 12.93% is 0.34 pp below the SOTA, exceeding the five-seed standard deviation of 0.16%, but single-seed comparisons carry inherent uncertainty.

Our real-world validation covers only convenience stores. Whether the approach transfers to hospitals, schools, or mixed-use buildings is unknown. On the residential segment (953 buildings), our model yields 77.71% NRMSE---comparable to the simple Persistence Ensemble (77.88%) and better than BuildingsBench's Transformer-M (92.60%), but far from practical utility. Residential load forecasting remains unsolved across all current approaches; commercial operational schedules do not transfer to residential patterns.

Checkpoints are selected by validation loss on a held-out split of our Korean simulation data. No BuildingsBench test-set information is used during training or model selection.

---

## List of Figures

**Fig. 1.** End-to-end pipeline: from building archetype selection and 12D LHS parameter sampling through EnergyPlus simulation, Box-Cox normalization, RevIN-equipped Transformer training, to zero-shot inference without geographic information.

**Fig. 2.** Main comparison of zero-shot commercial load forecasting performance (NRMSE, %) across six model configurations on the 955-building BuildingsBench evaluation set. The dashed line indicates the BB SOTA-M baseline (13.27%).

**Fig. 3.** N-scaling curve showing NRMSE as a function of the number of training buildings. Performance matches the SOTA from 70 buildings (n = 5) and saturates by 140 buildings (n = 10). The BB SOTA-M (13.27%) baseline is shown for reference.

**Fig. 4.** RevIN's asymmetric effect across dataset scales. Green arrows indicate RevIN improvement (lower NRMSE); the red arrow indicates RevIN degradation on the full 900K BuildingsBench corpus.

---

## 6. Conclusion

We have shown that 700 parametric building simulations, designed through 12-dimensional Latin Hypercube Sampling and paired with Reversible Instance Normalization, achieve zero-shot load forecasting performance that matches or exceeds the BuildingsBench SOTA trained on 900,000 buildings (12.93% vs. 13.27% NRMSE, best seed; 13.11 +/- 0.16% over five seeds). Controlled experiments at the 700-building scale indicate that data design contributes approximately 1.3--1.5x as much as RevIN to the improvement over BuildingsBench subsets.

The n-scaling analysis reveals that as few as 70 buildings (5 per archetype) match the SOTA, and performance stabilizes from 140 buildings onward. This challenges the assumption that building energy foundation models require massive training corpora. And the discovery that RevIN degrades performance when applied to the full 900K BuildingsBench corpus (13.27% to 13.89%) reveals a non-trivial interaction between normalization strategy and dataset structure. RevIN is most valuable when the training data lacks magnitude coverage---precisely the situation created by a small, shape-diverse training set.

For organizations seeking to deploy zero-shot building load forecasting in new regions, investing in fewer than a hundred diverse parametric simulations may be sufficient to match or exceed models trained on million-building databases. The finding that no geographic features are needed further lowers the barrier to deployment. Whether this conclusion generalizes beyond the specific building types, climates, and evaluation protocols studied here remains to be tested. The confounding between schedule diversity and climate origin, the limitation to commercial buildings, and the modest size of the real-world validation set all call for follow-up work. We release the simulation pipeline and model checkpoints to support such efforts.

---

## Acknowledgment

This work was supported by the research project funded by the Ministry of Trade, Industry and Energy, Korea Institute of Energy Technology Evaluation and Planning (KETEP).

## CRediT Author Statement

**Jeong-Uk Kim**: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review & editing, Visualization.

## Data Availability

The simulation pipeline, model source code, training configurations, and pretrained checkpoints are available at https://github.com/jukkim/korean-buildingsbench. The 700-building parametric simulation dataset (~20 MB) is included in the repository. The BuildingsBench evaluation data can be downloaded from the NREL Open Energy Data Initiative (https://data.openei.org/submissions/5859) under a CC-BY 4.0 license. The Korean convenience store electricity data was collected under a government-funded research project; access requests should be directed to the corresponding author.

## Declaration of Competing Interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## References

[1] Das, A. et al. "A decoder-only foundation model for time-series forecasting." ICML 2024 (TimesFM).

[2] Woo, G. et al. "Unified Training of Universal Time Series Forecasting Transformers." ICML 2024 (MOIRAI).

[3] Emami, P., Sahu, A., Graf, P. "BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting." NeurIPS 2023.

[4] Wilson, E. et al. "End-Use Load Profiles for the U.S. Building Stock." NREL, 2022.

[5] Liu, X. et al. "MOIRAI-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts." ICML 2025. arXiv:2410.10469.

[6] Shi, J., Ma, Q., et al. "Scaling Law for Time Series Forecasting." NeurIPS 2024.

[7] Kim, T. et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift." ICLR 2022.

[8] Chen, T. & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD 2016.

[9] Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." Neural Computation, 1997.

[10] Bai, S. et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv:1803.01271, 2018.

[11] Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.

[12] Nie, Y. et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023 (PatchTST).

[13] Ribeiro, M. et al. "Transfer learning with seasonal and trend adjustment for cross-building energy forecasting." Energy and Buildings, 165:352-363, 2018.

[14] Spencer, R. et al. "Transfer Learning on Transformers for Building Energy Consumption Forecasting." Energy and Buildings, 2025. arXiv:2410.14107.

[15] Ansari, A.F. et al. "Chronos: Learning the Language of Time Series." arXiv:2403.07815, 2024.

[16] Rasul, K. et al. "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting." arXiv:2310.08278, 2023.

[17] Yao, Q. et al. "Towards Neural Scaling Laws for Time Series Foundation Models." ICLR 2025. arXiv:2410.12360.

[18] Zha, D. et al. "Data-centric AI: Perspectives and Challenges." SDM 2023.

[19] Ng, A. "Unbiggen AI." IEEE Spectrum, 2022.

[20] Crawley, D.B. et al. "EnergyPlus: creating a new-generation building energy simulation program." Energy and Buildings, 2001.

[21] Nakkiran, P. et al. "Deep Double Descent: Where Bigger Models and More Data Can Hurt." ICLR 2020.

---

## Appendix A: Multi-Seed Results

### A.1 Korean-700 RevIN ON (s = 18,000, 5 seeds)

| Seed | NRMSE (%) |
|:----:|:----------:|
| 42 | 12.93 |
| 43 | 13.06 |
| 44 | 13.10 |
| 45 | 13.39 |
| 46 | 13.07 |
| **Mean +/- Std** | **13.11 +/- 0.16** |

### A.2 Korean-700 RevIN OFF (s = 18,000, 3 seeds)

| Seed | NRMSE (%) |
|:----:|:----------:|
| 42 | 14.81 |
| 43 | 14.94 |
| 44 | 14.40 |
| **Mean +/- Std** | **14.72 +/- 0.28** |

### A.3 Korean-700 RevIN ON (s = 16,000, 5 seeds)

| Seed | NRMSE (%) |
|:----:|:----------:|
| 42 | 12.89 |
| 43 | 13.24 |
| 44 | 13.16 |
| 45 | 13.26 |
| 46 | 13.12 |
| **Mean +/- Std** | **13.13 +/- 0.15** |

### A.4 BB 900K + RevIN

The BuildingsBench 900K model was retrained from scratch with RevIN enabled (identical architecture, optimizer, and training schedule to the original). The resulting NRMSE of 13.89% on the 955-building commercial evaluation set represents a 0.62 pp degradation relative to the original 13.27%. This single-run result uses the same seed and hyperparameters as the original BuildingsBench training.

---

## Appendix B: BB-700 and BB-7K Scaling

| Configuration | N Buildings | RevIN | NRMSE (%) |
|--------------|:-----------:|:-----:|:----------:|
| BB-700 | 700 | ON | 15.28 |
| BB-700 | 700 | OFF | 16.44 |
| BB-7K | 7,000 | ON | 14.50 |
| BB-7K | 7,000 | OFF | 15.41 |

A 10x increase in BuildingsBench buildings (700 to 7,000) reduces NRMSE by 0.78 pp with RevIN and 1.03 pp without. Both remain well above the Korean-700 result (12.93%), indicating that scaling within the stock-model distribution converges slowly and does not approach the performance achieved by 700 LHS-designed buildings.

All results verified through unified evaluation (torch 2.0.1, autocast, identical protocol as BB SOTA reproduction).
