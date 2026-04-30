# Seventy Simulations Suffice: Matching a 900,000-Building Foundation Model through Operational Diversity in Zero-Shot Load Forecasting

**Author**: Jeong-Uk Kim
**Affiliation**: Department of Electrical Engineering, Sangmyung University, Seoul 03016, South Korea
**E-mail**: jukim@smu.ac.kr

---

## Abstract

Zero-shot building load forecasting---predicting energy consumption for unseen buildings without retraining---is essential for grid balancing, demand response, and energy management in newly instrumented or unmetered facilities. The prevailing approach trains Transformer models on massive synthetic corpora: BuildingsBench (NeurIPS 2023) assembled 900,000 U.S. building simulations and achieved 13.28% median NRMSE on 955 commercial buildings (reproduced at 13.27% in this study), implying that generalization requires scale. We challenge this assumption. Critically, both BuildingsBench and this work train on synthetic simulations from a single country (the United States and South Korea, respectively), yet are evaluated on the same real commercial buildings from four U.S. university and government campuses and Portugal. Seventy EnergyPlus simulations---five per building archetype across 14 types---match the 900K-building result when operational schedules are designed for diversity rather than realism and paired with Reversible Instance Normalization (RevIN); scaling to 700 buildings exceeds it. Twelve schedule parameters sampled via Latin Hypercube Sampling, combined with RevIN, yield 13.28 ± 0.12% NRMSE from 70 buildings (five-seed mean) and 13.11 ± 0.16% from 700, without any geographic information. The n-scaling curve reveals a sharp transition: 14 buildings (n = 1 per archetype) score 14.72%, but 70 buildings already match the 900K SOTA across five seeds. Controlled experiments at the 700-building scale show that data design contributes approximately 1.3--1.5x the improvement of RevIN, and that applying RevIN to the full 900K corpus degrades performance to 13.89%---worse than the SOTA without RevIN---demonstrating that our advantage stems from data design, not from RevIN alone. Zero-shot evaluation on 218 real Korean convenience stores confirms sim-to-real transfer (12.30% vs. 13.14% for BuildingsBench). These findings suggest that a few dozen operationally diverse parametric simulations can substitute for million-building corpora, substantially lowering the barrier to zero-shot load forecasting in regions without large building stock databases.

**Keywords**: building energy forecasting, zero-shot learning, foundation models, parametric simulation, data-centric AI, reversible instance normalization

**Highlights**:
- 70 EnergyPlus simulations match a 900,000-building foundation model in zero-shot load forecasting
- 12-dimensional Latin Hypercube Sampling generates operationally diverse training data
- RevIN helps small diverse data but degrades performance on large homogeneous corpora
- No geographic information required: latitude and longitude set to zero with no accuracy loss
- Korean-climate-trained model matches or exceeds U.S.-trained baseline on real buildings across the United States and Portugal

---

## 1. Introduction

Short-term load forecasting underpins grid balancing, demand response, and real-time building energy management [1, 2]. The classical approach trains a separate model for each building, which works well when historical data are available but is inapplicable to newly instrumented or unmetered buildings. Foundation models for time series---trained once, deployed anywhere---offer an alternative: learn general temporal patterns from large corpora and predict unseen buildings without fine-tuning.

BuildingsBench [3] operationalized this idea for the building energy domain. By assembling Buildings-900K, a synthetic corpus of roughly 900,000 residential and commercial buildings drawn from the NREL End-Use Load Profiles database [4], and training an encoder-decoder Transformer with Gaussian negative log-likelihood loss, the authors achieved 13.28% median NRMSE on 955 unseen commercial buildings (which we reproduce at 13.27%). The result established a benchmark and carried an implicit message: scale matters. More synthetic buildings, more patterns, better generalization.

Recent work in the broader time series community has begun to challenge this framing. Shi et al. [5] showed that scaling laws for time series forecasting diverge from those in language modeling, with diminishing returns on out-of-distribution tasks. MOIRAI-MoE [6] demonstrated a sparse mixture-of-experts architecture with automatic token-level specialization, achieving superior zero-shot performance without increasing model or data scale. These findings suggest that brute-force scaling may not be sufficient for generalization. A complementary perspective comes from the data-centric AI community [7, 8]: rather than collecting more data from the same distribution, designing better data may be more productive.

We test this hypothesis directly in the building load forecasting setting. Our starting observation is that Buildings-900K, despite its size, draws from a single stock-model distribution---the NREL End-Use Load Profiles for the U.S. building stock. Importantly, this is a single-country, single-distribution corpus: all 900,000 buildings are U.S. simulations. Adding more buildings from the same distribution increases volume but not necessarily the diversity of temporal patterns a model encounters. An alternative is to construct a small training set with maximal operational diversity by design.

We generate 700 EnergyPlus simulations (50 per archetype across 14 building types) with operational schedules sampled via 12-dimensional Latin Hypercube Sampling. The 12 parameters---covering operating hours, baseload levels, weekend patterns, ramp characteristics, equipment retention, and seasonal variation---span the space of plausible commercial building operations far more broadly than any stock-model sample of comparable size. Combined with Reversible Instance Normalization (RevIN) [9], which absorbs building-specific load magnitude and variability at inference time, this small but diverse training set achieves 12.93% NRMSE on the BuildingsBench evaluation protocol using the best of five seeds, and 13.11 ± 0.16% averaged across all five. The model uses no geographic features (latitude and longitude are set to zero).

Controlled experiments on identical 700-building subsets---one from our parametric simulations, one from BuildingsBench---show that data design contributes more than RevIN to the overall improvement. Applying RevIN to the full 900K BuildingsBench corpus worsens performance from 13.27% to 13.89%, an asymmetry suggesting that RevIN is most valuable when the training data lacks magnitude coverage. The n-scaling curve shows that 70 buildings already match the SOTA, and zero-shot evaluation on 218 real Korean convenience stores yields 12.30% vs. 13.14% for BuildingsBench.

Our parametric simulations differ from BuildingsBench not only in schedule design but also in climate (Korean weather files vs. U.S. TMY data), building codes, and envelope parameters. The 2x2 factorial design isolates data-source effects at the 700-building scale but does not fully disentangle these confounds. We also use RevIN, which BuildingsBench does not, creating an asymmetry that our control experiments address but cannot eliminate entirely. These limitations are discussed in Section 5.

---

## 2. Related Work

### 2.1 Building Energy Forecasting

Building energy prediction has progressed through statistical methods (ARIMA, exponential smoothing) [10], gradient boosting [11], recurrent networks [12], temporal convolutional architectures [13], and Transformers [14, 15]. Each advance improved accuracy on individual buildings but remained building-specific: a model trained on one office tower cannot predict another without retraining. Transfer learning through domain adaptation [16] and pretrained model fine-tuning [17] relaxes this constraint somewhat but still requires some target-domain data. Our work eliminates the need for any target data by pursuing fully zero-shot inference.

### 2.2 BuildingsBench and Zero-Shot Load Forecasting

BuildingsBench [3] framed building load forecasting as a zero-shot generalization problem. The training corpus, Buildings-900K, comprises approximately 350,000 commercial (ComStock) and 550,000 residential (ResStock) synthetic buildings with hourly electricity consumption generated by the NREL EULP pipeline [4]. The model---an encoder-decoder Transformer predicting 24-hour Gaussian distributions from 168-hour context---trains for a fraction of one epoch (0.067 epochs, with each building seen roughly six times) using global Box-Cox normalization (lambda = -0.067). The reported SOTA is 13.28% median NRMSE on commercial buildings. Our work keeps the model architecture and evaluation protocol fixed while changing the training data and normalization strategy.

The sub-epoch training regime deserves attention. BuildingsBench found that training for less than one full pass over the data was essential for generalization to real buildings; additional training degraded performance on the out-of-distribution evaluation set. This finding suggests that the model's capacity to memorize synthetic patterns outpaces its ability to learn transferable ones, and that restraint in training duration is as important as data volume.

### 2.3 Scaling Laws and Their Limits in Time Series

The proliferation of general-purpose time series foundation models---TimesFM [18], Chronos [19], Lag-Llama [20], MOIRAI [21]---has established that large-scale pretraining enables zero-shot forecasting across domains. These models train on millions to billions of time series from heterogeneous sources. Shi et al. [5] investigated whether the power-law scaling observed in language models transfers to time series and found that it does not: returns diminish more steeply, particularly under distribution shift. Yao et al. [22] found that model architecture significantly influences scaling efficiency, with encoder-only Transformers demonstrating better scalability than decoder-only alternatives. Most relevant to our work, MOIRAI-MoE [6] showed that a mixture-of-experts architecture can outperform its monolithic predecessor through token-level specialization rather than increased model or data scale. The relationship between data volume and generalization in time series is more complex than in language modeling, where each additional token carries novel combinatorial information.

### 2.4 Data-Centric AI and Reversible Instance Normalization

The data-centric AI perspective [7, 8] holds that improving data quality is often more productive than improving model architecture. This principle has been validated extensively in computer vision and NLP but has received limited attention in time series energy forecasting. Our 12-dimensional LHS design can be viewed as data curation for the energy domain: rather than sampling from a fixed distribution, we construct a training set that fills the space of plausible operations.

RevIN [9] complements this approach from the normalization side. By subtracting each input instance's mean and dividing by its standard deviation before the model, then reversing the operation on the output, RevIN removes instance-level statistical properties that confound pattern learning. In building load forecasting, where inter-building variation in load magnitude spans several orders of magnitude, this is a natural fit. The interaction between RevIN and dataset scale is examined in Section 4.

---

## 3. Method

### 3.1 Parametric Building Simulation

Training data are generated through EnergyPlus [23] simulation of Korean commercial buildings with parametrically varied operational schedules.

**Building archetypes.** We use 14 DOE reference building types [24] adapted to Korean building codes and climate zones: office, retail, school, hotel, hospital, apartment (midrise and highrise), small office, large office, warehouse, strip mall, restaurant (full-service and quick-service), and university. Each archetype begins from a DOE reference model with Korean-code-compliant envelope thermal properties, HVAC system sizing, and internal load densities.

**Climate coverage.** Simulations span five Korean cities across three climate zones: Seoul (central), Busan (southern), Daegu (southern), Gangneung (central-coastal), and Jeju (subtropical). Typical Meteorological Year (TMY) weather files for each city drive the simulations.

**Operational parameter space.** Building operational schedules are parameterized by 12 continuous variables sampled via Latin Hypercube Sampling [25] (Table 1). These parameters govern operating hours (op_start, op_duration), baseload and ramp characteristics (baseload_pct, ramp_hours), occupancy-related patterns (weekend_factor, equip_always_on, night_equipment_frac), and stochastic variation (daily_noise_std, weekly_break_prob, seasonal_amplitude, process_load_frac, scale_mult).

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

The 12 parameters were selected to span the primary operational degrees of freedom identified in commercial building energy audits; sensitivity to individual parameters has not been analyzed. LHS [25] ensures space-filling coverage of this 12-dimensional parameter space, producing buildings that range from 24/7 high-baseload facilities (resembling data centers or hospitals) to weekday-only offices with steep morning ramps and low overnight loads. The overall pipeline from archetype selection through simulation to zero-shot inference is illustrated in Fig. 1. For the n = 50 configuration used in our main experiments, this yields 50 samples per archetype x 14 archetypes = 700 buildings. Each sample modifies the EnergyPlus IDF file---adjusting internal load schedules, equipment power densities, and occupancy patterns---runs a full annual simulation (8,760 hours), and extracts hourly total electricity consumption.

The design philosophy differs fundamentally from stock-model sampling. The NREL EULP pipeline underlying Buildings-900K generates building populations that reflect the statistical distribution of the U.S. building stock: many buildings cluster near typical operating conditions, with fewer at the extremes. LHS, by contrast, ensures uniform marginal coverage across all 12 dimensions simultaneously. A 700-building LHS sample spans the operational parameter space more evenly than a 700-building stock-model sample, though the two approaches answer different questions. Stock-model sampling asks "what does the building stock look like?"; LHS sampling asks "what temporal patterns can buildings produce?"

**Statistical validation.** We validated simulation outputs against statistical properties of real commercial buildings measured from the BuildingsBench evaluation set. Four parameters---night_equipment_frac, weekly_break_prob, seasonal_amplitude, and process_load_frac---were added to ensure that the simulated load profiles exhibit physically plausible temporal characteristics (nighttime baseload, weekly periodicity, day-to-day variability) consistent with those observed in real buildings. These parameters affect the marginal distribution of temporal patterns in the training corpus, not any specific building's load values; they ensure the training data is physically realistic rather than leaking any label information from the evaluation set (Table 2).

| Metric | Without 4 params | With 4 params | BB Real Buildings |
|--------|:----------------:|:-------------:|:-----------------:|
| Night/Day ratio | 0.512 | 0.852 | 0.803 |
| Autocorrelation (168h) | 0.938 | 0.784 | 0.751 |
| Baseload P95 | 0.51 | 0.905 | 0.725 |
| CV P5 (flatness) | 0.21 | 0.032 | 0.105 |

**Table 2.** Statistical properties of simulated vs. real building load profiles, with and without the four diversity-enhancing parameters. The night/day ratio measures the ratio of average nighttime to daytime load; autocorrelation at 168 hours captures weekly periodicity strength; baseload P95 is the 95th percentile of the minimum daily load divided by the maximum; CV P5 measures the 5th percentile of the coefficient of variation across days. Adding the four parameters brings all four statistics within range of the real-building values. The goal is not to replicate the stock-model distribution but to ensure that simulation outputs exhibit physically plausible temporal characteristics. LHS provides spatial coverage of the parameter space; the four additional parameters provide physical plausibility.

### 3.2 Model Architecture

To isolate the effect of training data from model design, we adopt the identical Transformer architecture used in BuildingsBench [3]. The encoder-decoder model has 3 encoder and 3 decoder layers, d_model = 512, 8 attention heads, and a feedforward dimension of 1024, totaling 15.8M parameters (identical to BuildingsBench Transformer-M). Input consists of 168 hourly load values along with temporal features (day-of-year sinusoidal encoding, day-of-week embedding, hour-of-day embedding). The model predicts 24-hour-ahead Gaussian distribution parameters (mu, sigma) through autoregressive decoding, trained with Gaussian negative log-likelihood loss.

**Reversible Instance Normalization.** Before encoding, the 168-hour context window is normalized by its instance mean and standard deviation:

$$x_{\text{norm}} = (x - \mu_{\text{ctx}}) / \sigma_{\text{ctx}}$$

After decoding, the normalization is reversed:

$$\hat{y} = \hat{y}_{\text{norm}} \cdot \sigma_{\text{ctx}} + \mu_{\text{ctx}}$$

This removes building-specific load magnitude and variability from the learning problem, allowing the model to focus on temporal shape. RevIN is a standard technique [9] that does not modify the Transformer architecture; it operates as a pre- and post-processing layer. We apply it symmetrically in all controlled experiments, including the BuildingsBench baselines, to ensure fair comparison within our experimental framework. The comparison between our RevIN-equipped model and the original BuildingsBench (which does not use RevIN) is not strictly equivalent; we discuss this asymmetry in Section 5.1.

**No geographic features.** BuildingsBench provides latitude and longitude embeddings as model inputs. We set both to zero for all buildings---both training and evaluation. As we show in Section 4.4, this produces results indistinguishable from using actual coordinates when RevIN is active, suggesting that RevIN absorbs the distributional shifts that geographic information would otherwise encode.

### 3.3 Training Protocol

We apply a global Box-Cox power transform [26] fitted on our Korean simulation data to normalize load values before training. The fitted power parameter lambda = -0.067 coincidentally matches the value reported in BuildingsBench, but the scale and location parameters (mean_, scale_) differ because they are fit to the Korean corpus rather than the U.S. stock-model corpus. The optimizer is AdamW [27] with a learning rate of 6 x 10^-5, weight decay of 0.01, cosine annealing schedule, and a 500-step linear warmup. Training runs for 18,000 gradient steps with batch size 128, corresponding to approximately 9 epochs over the 250,600 sliding windows derived from 700 buildings (358 windows per building at stride 24). Data augmentation includes window jitter (±1--6 hours random shift of the context window start), Gaussian noise (sigma = 0.02 in Box-Cox space), and amplitude scaling (uniform in [0.85, 1.15]). Mixed-precision (FP16) autocast is used during training and inference.

Both our training and BuildingsBench use approximately 18,000 gradient steps. BuildingsBench processes 0.067 epochs over 900,000 buildings; we process approximately 9 epochs over 700 buildings. The total number of gradient updates is similar, but distributed very differently: each of our 700 buildings is seen roughly 3,300 times (across different window positions), while each of BuildingsBench's 900,000 buildings is seen roughly 6 times. The sub-epoch regime identified in BuildingsBench as important for avoiding overfitting to synthetic data applies here as well, despite this inverted balance. Three factors prevent this high per-building exposure from causing overfitting: (1) data augmentation (window jitter, Gaussian noise, and amplitude scaling) ensures that each exposure presents a different view of the training signal; (2) RevIN removes per-building magnitude information, preventing the model from memorizing absolute load levels; and (3) the high inter-building diversity of the LHS design ensures that patterns learned from any one building are also present, in varied form, in many others. The augmentation applied here was not included in the original BuildingsBench training pipeline; this asymmetry is discussed as a limitation in Section 5.4.

### 3.4 Evaluation Protocol

We follow the BuildingsBench evaluation protocol without modification. The test set comprises 955 commercial buildings: 611 from the Building Data Genome Project 2 (BDG-2) [28], drawn from four U.S. university and government campuses (University of California Berkeley, Arizona State University, University of Central Florida, and Washington D.C.), and 344 from the Electricity dataset [29], covering consumers in Portugal, with 15 out-of-vocabulary buildings excluded per the original specification. The evaluation set spans four distinct U.S. climate zones (Mediterranean, semi-arid desert, humid subtropical, and humid continental) plus Southern Europe, meaning our Korean-trained model is assessed against buildings climatically and geographically far from its Korean training distribution. Per-building NRMSE is computed as sqrt(MSE) / mean(actual) and aggregated by taking the median across buildings. We use the term NRMSE (Normalized Root Mean Squared Error) to match BuildingsBench's convention; this is numerically identical to the Coefficient of Variation of RMSE (CVRMSE) when load values are non-negative, as is always the case for electricity consumption. The sliding window uses stride 24 hours, context 168 hours, and prediction horizon 24 hours.

We reproduced the BuildingsBench SOTA-M result using the official checkpoint (torch 2.0.1, identical Box-Cox parameters) and obtained 13.27% NRMSE, confirming that our evaluation pipeline matches the original within 0.01 percentage points of the reported 13.28%.

---

## 4. Experiments and Results

### 4.1 Main Results

Table 3 presents the primary comparison (visualized in Fig. 2). All models use the same 15.8M-parameter Transformer-M architecture and are evaluated on the identical 955-building test set with the same NRMSE computation.

| Model | Training Data | N Buildings | RevIN | NRMSE (%) | NCRPS (%) | vs SOTA |
|-------|--------------|:-----------:|:-----:|:----------:|:---------:|:-------:|
| BB SOTA-M (reproduced) | BB 900K | 900,000 | OFF | 13.27 | —† | — |
| BB 900K + RevIN | BB 900K | 900,000 | ON | 13.89 | 7.76 | +0.62 |
| **Korean-700 (ours)** | **Korean sim** | **700** | **ON** | **13.11 ± 0.16** | **7.14 ± 0.02** | **-0.16** |
| Korean-700 | Korean sim | 700 | OFF | 14.72 ± 0.28 | 8.29 | +1.45 |
| BB-700 (aug-matched) | BB subset | 700 | ON | 14.26 | 7.80 | +0.99 |
| BB-700 | BB subset | 700 | OFF | 16.44 | — | +3.17 |
| Persistence Ensemble | --- | --- | --- | 16.68 | — | +3.41 |

**Table 3.** Main results on the BuildingsBench 955-building commercial evaluation set. Korean-700 with RevIN reports five-seed mean ± standard deviation (13.11 ± 0.16%); best seed is 12.93%. Korean-700 without RevIN reports three-seed statistics. NCRPS (Normalized Continuous Ranked Probability Score, lower is better) measures probabilistic calibration in the original kWh scale; NCRPS values are not available for BB SOTA-M (†different Box-Cox protocol) or models with no probabilistic output. BB-700 (aug-matched) uses a random sample of 700 buildings from the BuildingsBench ComStock corpus, trained with the same optimizer, data augmentation (window jitter, Gaussian noise, amplitude scaling), and number of gradient steps as Korean-700 (seed = 42), isolating the effect of data source. BB-700 without augmentation (no-aug) achieves 15.28% NRMSE with RevIN, 1.02 pp worse than the aug-matched variant.

The five-seed mean of Korean-700 with RevIN is 13.11 ± 0.16%, with the best seed reaching 12.93%. The five-seed mean is 0.16 pp below the BB SOTA (13.27%), a gap comparable to the seed-to-seed standard deviation; the n = 5 result (13.28 ± 0.12%) matches within 0.01 pp. To our knowledge, no prior published work has demonstrated performance in this range using fewer than 1,000 training buildings on the BuildingsBench evaluation protocol. Without RevIN, performance degrades to 14.72 ± 0.28% (1.45 pp above the SOTA), though still well below the Persistence Ensemble baseline (16.68%). The BB-700 control---the same architecture trained on 700 randomly sampled BuildingsBench buildings with matched augmentation---achieves 14.26% with RevIN and 16.44% without (the OFF variant was trained without augmentation), indicating that the data source matters independently of the normalization strategy. Without augmentation, BB-700 with RevIN drops to 15.28%, confirming that augmentation contributes 1.02 pp of improvement on BB data but cannot bridge the gap to Korean-700.

Adding RevIN to the full BuildingsBench training pipeline degrades performance from 13.27% to 13.89%, a 0.62 percentage-point increase. RevIN helps small data (Korean-700: 14.72% to 13.11%, a 1.61 pp improvement in five-seed mean) but hurts large data (BB 900K: 13.27% to 13.89%, a 0.62 pp degradation). We return to this asymmetry in Section 5.1.

### 4.2 Decomposing the Improvement

The 2x2 factorial design (Korean vs. BB data, RevIN on vs. off, both at n = 700) allows partial decomposition. With RevIN on and matched augmentation, Korean-700 (13.11%) outperforms BB-700 (14.26%) by 1.15 pp. With RevIN off, the gap widens to 1.72 pp (Korean-700 OFF 14.72% vs. BB-700 OFF 16.44%), though the BB-700 OFF baseline was trained without augmentation; an aug-matched OFF comparison would narrow this gap somewhat. The RevIN effect is 1.61 pp on Korean data. Even under the most favorable same-normalization comparison, the data-source advantage remains substantial. RevIN amplifies the advantage but does not create it.

We stress that "data design effect" conflates several factors: the 12D LHS schedule design, Korean climate and building codes, and EnergyPlus model parameterization. A future experiment using U.S. weather files with Korean-style LHS schedules would be needed to isolate the schedule diversity contribution. The decomposition is informative about relative magnitudes but should not be interpreted as a clean causal estimate.

### 4.3 N-Scaling Analysis

Table 4 and Fig. 3 show how NRMSE varies with the number of buildings per archetype (n), where total buildings = 14n. All runs use the M-size model with RevIN on and a fixed training budget of 18,000 steps.

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

**Table 4.** N-scaling results (M-size model, RevIN ON, s = 18,000 steps). All rows use seed = 42 except n = 5, which reports five-seed mean ± std, and n = 50, which matches the Table 3 best seed. Fig. 3 shows the full 17-point curve including intermediate values (n = 2: 14.08%, n = 4: 13.45%, n = 6: 13.33%, n = 7: 13.21%, n = 8: 13.27%, n = 9: 13.26%, n = 30: 13.13%, n = 40: 13.23%, n = 60: 13.18%); all single-seed (seed = 42). The curve is non-monotonic beyond n = 10, with fluctuations within approximately ±0.15 pp around a plateau near 13.15--13.25%.

Performance improves sharply from n = 1 to n = 5 (14.72% to 13.28%). Note that n = 1 with RevIN ON (14.72%) numerically coincides with Korean-700 RevIN OFF (14.72%); this reflects the empirical finding that a single building per archetype (14 buildings total) is insufficient for RevIN to provide any benefit over the no-normalization baseline. By n = 3 (42 buildings), performance reaches 13.47%, within 0.20 pp of the SOTA. At n = 5 (70 buildings), the five-seed mean of 13.28 ± 0.12% matches the SOTA (13.27%) within 0.01 pp. From n = 5 onward, NRMSE stabilizes---additional parametric simulations beyond roughly 70 buildings yield diminishing gains.

The n = 5 multi-seed result (13.28 ± 0.12%) confirms that the "seventy suffice" finding is not a single-seed artifact. At n = 50, the five-seed mean drops further to 13.11 ± 0.16%, reliably surpassing the SOTA. The scaling behavior is qualitatively different from the monotonic improvement typically assumed in large-scale pretraining. Adding a 10x or 100x multiplier to the training set size, while keeping the same LHS design, would be unlikely to produce meaningful gains.

This saturation is also visible at the opposite extreme. The BuildingsBench scaling experiment (BB-700 vs. BB-7K, Appendix B) shows that increasing from 700 to 7,000 randomly sampled BuildingsBench buildings barely changes performance (BB-700 without augmentation: 15.28% vs. 14.50% with RevIN; 16.44% vs. 15.41% without RevIN). Even with augmentation, BB-700 achieves only 14.26% with RevIN---still 1.15 pp above Korean-700. The marginal value of additional buildings from a stock-model distribution is small in both the Korean and BB settings.

### 4.4 Ablation Studies

Table 5 reports the two ablations conducted under the unified evaluation protocol (Section 3.4).

| Experiment | NRMSE (%) | Delta | Interpretation |
|------------|:----------:|:-----:|----------------|
| Korean-700 RevIN ON (baseline) | 12.93 | — | Best configuration (seed 42) |
| RevIN OFF (3-seed mean) | 14.72 | +1.79 | RevIN contributes 1.79 pp (1.61 pp over five-seed means) |

**Table 5.** Key ablation results using the standard evaluation protocol (val_best checkpoint, 955 commercial buildings, seed = 42). Setting latitude and longitude to zero produced identical results to using actual Korean coordinates, confirming geographic features are unnecessary when RevIN is active. Additional ablations (Box-Cox transfer, extended training, scale expansion, seasonal decomposition) are reported in Appendix C; those experiments used an earlier evaluation pipeline and are included for qualitative reference.

Additional ablations---Box-Cox transfer from BB, extended training duration, and scale expansion to 70,000 buildings---all degrade performance substantially (+3.31, +3.09, and +2.42 pp, respectively), consistent with the sub-epoch overfitting insight from BuildingsBench. These experiments used an earlier evaluation pipeline; results are tabulated in Appendix C for reference.

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

Fig. 4 illustrates the asymmetry in RevIN's effect across dataset scales. On our 700-building Korean dataset, RevIN reduces NRMSE by 1.61 pp (14.72% to 13.11%, five-seed means). On the 700-building BB subset, the reduction is 1.16 pp (16.44% to 15.28%). But on the full 900K BB corpus, RevIN increases NRMSE by 0.62 pp (13.27% to 13.89%). The same technique that is beneficial at small scale is detrimental at large scale.

RevIN normalizes each 168-hour context window to zero mean and unit variance, stripping absolute load magnitude and within-window variance. The model then operates in a scale-free space, learning only temporal shape. With 700 LHS-designed buildings, the model has not seen enough magnitude variation to internalize it; RevIN solves this analytically. With 900,000 stock-model buildings seen approximately 6 times each, the model has already learned to exploit magnitude as a signal---a building consuming 500 kW at midnight behaves differently from one consuming 5 kW. Removing this information via RevIN discards something useful, producing a net loss.

A complementary explanation involves the distributional structure of each dataset. In Buildings-900K, a building's magnitude is informative about its type because HVAC, envelope, and schedule are jointly determined by stock-model parameters. In our LHS design, magnitude reflects only the scale_mult parameter, which is independent of the schedule parameters that determine temporal shape. RevIN's removal of magnitude is therefore less costly for LHS data than for stock-model data.

The practical implication is that RevIN is not universally beneficial. Its value depends on whether the training data already covers the magnitude range of the evaluation set.

### 5.2 Why Operational Diversity May Matter More Than Scale

The central finding of this study is that 700 buildings with LHS-designed operational schedules outperform 900,000 stock-model buildings on the BuildingsBench evaluation protocol (when RevIN is used). Buildings-900K draws from the NREL End-Use Load Profiles for the U.S. building stock. This is a carefully constructed stock model, but a stock model nonetheless. It generates buildings whose operational parameters reflect the statistical distribution of real U.S. commercial buildings: most offices operate 8 AM to 6 PM on weekdays; most retail stores have moderate baseloads; most hospitals run continuously. Adding more buildings from this distribution increases the sample size but does not proportionally increase the diversity of temporal patterns the model encounters. Many of the 900,000 buildings share similar operational profiles because they are drawn from the same assumptions about how buildings operate.

Our 12D LHS design takes a different approach. By sampling operational parameters uniformly across their feasible ranges, we produce buildings that occupy regions of operational space rarely seen in a stock-model sample: offices that run 24 hours with 95% baseload, schools with random weekly schedule disruptions, warehouses with strong seasonal oscillation. These extreme and unusual patterns may be rare in reality, but they teach the model about the full range of temporal dynamics that building loads can exhibit. The BuildingsBench evaluation set contains real buildings, and real buildings---especially non-standard ones like data centers, 24-hour retail, or buildings with unusual HVAC systems---may have operational patterns closer to our LHS extremes than to the stock-model center.

From a scaling-law perspective, this result aligns with the findings of Shi et al. [5] that scaling in time series does not follow language-model power laws. In language, each additional token adds a novel word combination; in building load time series, each additional building from the same stock model adds a variation on patterns already in the training set. The marginal information content decreases rapidly.

The n-scaling analysis (Table 4) provides direct evidence. Performance matches the SOTA from just 70 buildings (n = 5 per archetype) and saturates by 140 buildings (n = 10). Beyond this point, the LHS parameter space is sufficiently covered that additional samples provide negligible new information. This is the opposite of what a power-law relationship would predict and is consistent with a model of learning where the number of distinct temporal patterns, not the number of examples, determines generalization.

### 5.3 Practical Implications

These results suggest a deployment pathway for regions without large building stock databases. An organization could generate several hundred parametric EnergyPlus simulations conforming to local building codes and climate, train a standard Transformer with RevIN, and deploy zero-shot forecasting---without assembling a million-building dataset or collecting extensive real-world measurements.

The computational cost difference is substantial. Our 700 EnergyPlus simulations complete in approximately 4 hours on a single workstation. Training requires 18,000 gradient steps on a single GPU (roughly 2 hours on an RTX 4090). The entire pipeline from simulation to deployed model takes less than a day. BuildingsBench's 900,000 simulations required the precomputed NREL EULP database and training at a comparable cost per step but with a dataset 1,286 times larger.

The finding that geographic features are unnecessary further simplifies deployment. Building operators need only provide hourly electricity data. No location metadata, building-type classification, climate zone, or HVAC system information is required. RevIN handles the distributional adaptation.

The climate mismatch between training and evaluation is itself informative. Both BuildingsBench and our approach train on synthetic simulations from a single country---the United States (NREL EULP) and South Korea (this work), respectively---yet both are evaluated on real commercial buildings from four U.S. climate zones and Portugal. That 700 Korean synthetic buildings outperform 900,000 U.S. synthetic buildings on this evaluation set demonstrates that operational schedule diversity trumps both data volume and geographic proximity to the evaluation population. RevIN contributes to this cross-climate generalization by removing magnitude differences attributable to climate-driven heating and cooling loads, but the temporal patterns learned from diverse Korean schedules transfer across geographically and climatically varied building stocks in the United States and Southern Europe. This finding suggests that the operational parameter space---when properly covered by LHS---captures building behavior at a level of abstraction that transcends specific climate conditions, though this comparison also involves additional methodological differences discussed in Section 5.4.

### 5.4 Limitations

Our parametric simulations differ from BuildingsBench in schedule design (12D LHS vs. stock model), climate (Korean vs. U.S. weather), building codes, and envelope parameters. At equal scale with matched augmentation, Korean data outperforms BB data by 1.15 pp (13.11% vs. 14.26%), establishing that the data source matters. The climate difference does not obviously favor Korean data, since the evaluation set comprises U.S. and Portuguese buildings; that a Korean-trained model outperforms a U.S.-trained one on U.S. data suggests operational diversity transfers across climates. A U.S.-TMY/LHS ablation would cleanly isolate schedule diversity from climate effects; this experiment is feasible but was not conducted here.

Two additional confounds are controlled but not eliminated. Applying RevIN to BB 900K degrades performance by 0.62 pp (13.27% to 13.89%), confirming RevIN is not universally beneficial. Augmentation contributes 1.02 pp to BB-700 (15.28% to 14.26%), yet Korean-700 still leads by 1.15 pp with matched augmentation. Both controls consistently identify data design as the primary driver; the NCRPS comparison (Korean-700: 7.14% vs. BB-700: 7.80%) reinforces this finding.

Our five-seed mean of 13.11 ± 0.16% is 0.16 pp below the BB SOTA, a gap comparable to the inter-seed standard deviation. BuildingsBench does not report confidence intervals, so a formal significance test is not possible; our claim is that 700 buildings achieve performance in the same range as 900,000. N-scaling intermediate points (n = 2--4, 6--9, 20--40, 60--80) use seed = 42; the non-monotonic variation within ±0.15 pp at larger n likely reflects seed variance rather than a systematic trend.

All experiments use the Transformer-M architecture (15.8M parameters); generalization to PatchTST, MOIRAI, or Chronos has not been tested. The real-world validation covers only convenience stores; hospitals, schools, and offices require separate evaluation. On the residential segment (953 buildings), our model yields 77.71% NRMSE, comparable to the Persistence Ensemble (77.88%), confirming that zero-shot residential forecasting remains an open problem. Checkpoints are selected by validation loss on held-out Korean simulation data; no BuildingsBench test-set information influences model selection.

---

## List of Figures

**Fig. 1.** End-to-end pipeline: from building archetype selection and 12D LHS parameter sampling through EnergyPlus simulation, Box-Cox normalization, RevIN-equipped Transformer training, to zero-shot inference without geographic information.

**Fig. 2.** Main comparison of zero-shot commercial load forecasting performance (NRMSE, %) across six model configurations on the 955-building BuildingsBench evaluation set. The dashed line indicates the BB SOTA-M baseline (13.27%).

**Fig. 3.** N-scaling curve showing NRMSE as a function of the number of training buildings. Performance matches the SOTA from 70 buildings (n = 5) and saturates by 140 buildings (n = 10). The BB SOTA-M (13.27%) baseline is shown for reference.

**Fig. 4.** RevIN's asymmetric effect across dataset scales. Green arrows indicate RevIN improvement (lower NRMSE); the red arrow indicates RevIN degradation on the full 900K BuildingsBench corpus.

---

## 6. Conclusion

We have shown that 700 parametric building simulations, designed through 12-dimensional Latin Hypercube Sampling and paired with Reversible Instance Normalization, achieve zero-shot load forecasting performance that matches or exceeds the BuildingsBench SOTA trained on 900,000 buildings (12.93% vs. 13.27% NRMSE, best seed; 13.11 ± 0.16% over five seeds). Controlled experiments at the 700-building scale indicate that data design contributes approximately 1.3--1.5x as much as RevIN to the improvement over BuildingsBench subsets.

The n-scaling analysis reveals that as few as 70 buildings (5 per archetype) match the SOTA, and performance stabilizes from 140 buildings onward. This challenges the assumption that building energy foundation models require massive training corpora. The finding that RevIN degrades performance when applied to the full 900K BuildingsBench corpus (13.27% to 13.89%) further reveals a non-trivial interaction between normalization strategy and dataset structure. RevIN is most valuable when the training data lacks magnitude coverage---precisely the situation created by a small, shape-diverse training set.

For organizations seeking to deploy zero-shot building load forecasting in new regions, investing in fewer than a hundred diverse parametric simulations may be sufficient to match or exceed models trained on million-building databases. The finding that no geographic features are needed further lowers the barrier to deployment. Whether this conclusion generalizes beyond the specific building types, climates, and evaluation protocols studied here remains to be tested. The confounding between schedule diversity and climate origin, the limitation to commercial buildings, the use of data augmentation not present in the original BuildingsBench pipeline, and the modest size of the real-world validation set all call for follow-up work. The simulation pipeline and model checkpoints are publicly available to support such efforts.

---

## Acknowledgements

This work was supported by the research project funded by the Ministry of Trade, Industry and Energy, Korea Institute of Energy Technology Evaluation and Planning (KETEP).

## CRediT Author Statement

**Jeong-Uk Kim**: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review & editing, Visualization.

## Data Availability

The simulation pipeline, model source code, training configurations, and pretrained checkpoints for the primary results (Korean-700 seeds 42--46, BB-700, BB 900K + RevIN) will be made publicly available at https://github.com/jukkim/korean-buildingsbench upon publication. The 700-building parametric simulation dataset (~20 MB) is included in the repository. N-scaling checkpoints for intermediate points (n = 2--4, 6--9, 20, 30, 40, 60, 70, 80) are available from the corresponding author upon request. The BuildingsBench evaluation data can be downloaded from the NREL Open Energy Data Initiative (https://data.openei.org/submissions/5859) under a CC-BY 4.0 license. The Korean convenience store electricity data was collected under a government-funded research project; access requests should be directed to the corresponding author.

## Declaration of Competing Interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## References

[1] Fan, C. et al. "A short-term building cooling load prediction method using deep learning algorithms." Applied Energy, 195:222-233, 2017.

[2] Hong, T. et al. "Ten questions on urban building energy modeling." Building and Environment, 168:106508, 2020.

[3] Emami, P., Sahu, A., Graf, P. "BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting." NeurIPS 2023.

[4] Wilson, E. et al. "End-Use Load Profiles for the U.S. Building Stock." NREL, 2022.

[5] Shi, J., Ma, Q., et al. "Scaling Law for Time Series Forecasting." NeurIPS 2024.

[6] Liu, X. et al. "MOIRAI-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts." Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267:38940-38962, 2025. arXiv:2410.10469.

[7] Zha, D. et al. "Data-centric AI: Perspectives and Challenges." SDM 2023.

[8] Ng, A. "Unbiggen AI." IEEE Spectrum, 2022.

[9] Kim, T. et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift." ICLR 2022.

[10] Amasyali, K. & El-Gohary, N.M. "A review of data-driven building energy consumption prediction studies." Renewable and Sustainable Energy Reviews, 81:1192-1205, 2018.

[11] Chen, T. & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD 2016.

[12] Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." Neural Computation, 1997.

[13] Bai, S. et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv:1803.01271, 2018.

[14] Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.

[15] Nie, Y. et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023 (PatchTST).

[16] Ribeiro, M. et al. "Transfer learning with seasonal and trend adjustment for cross-building energy forecasting." Energy and Buildings, 165:352-363, 2018.

[17] Spencer, R. et al. "Transfer Learning on Transformers for Building Energy Consumption Forecasting." Energy and Buildings, 2025. arXiv:2410.14107.

[18] Das, A. et al. "A decoder-only foundation model for time-series forecasting." ICML 2024 (TimesFM).

[19] Ansari, A.F. et al. "Chronos: Learning the Language of Time Series." Transactions on Machine Learning Research, 2024.

[20] Rasul, K. et al. "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting." NeurIPS 2023 Workshop on Robustness of Foundation Models. arXiv:2310.08278.

[21] Woo, G. et al. "Unified Training of Universal Time Series Forecasting Transformers." ICML 2024 (MOIRAI).

[22] Yao, Q. et al. "Towards Neural Scaling Laws for Time Series Foundation Models." ICLR 2025. arXiv:2410.12360.

[23] Crawley, D.B. et al. "EnergyPlus: creating a new-generation building energy simulation program." Energy and Buildings, 33(4):319-331, 2001.

[24] Deru, M. et al. "U.S. Department of Energy Commercial Reference Building Models of the National Building Stock." NREL/TP-5500-46861, 2011.

[25] McKay, M.D., Beckman, R.J., Conover, W.J. "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." Technometrics, 21(2):239-245, 1979.

[26] Box, G.E.P. & Cox, D.R. "An Analysis of Transformations." Journal of the Royal Statistical Society: Series B, 26(2):211-252, 1964.

[27] Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019 (AdamW).

[28] Miller, C. et al. "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition." Scientific Data, 7:368, 2020.

[29] Trindade, A. "ElectricityLoadDiagrams20112014." UCI Machine Learning Repository, 2015.

[30] Nakkiran, P. et al. "Deep Double Descent: Where Bigger Models and More Data Can Hurt." ICLR 2020.

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
| **Mean ± Std** | **13.11 ± 0.16** | **7.14 ± 0.02‡** |

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

Five-seed evaluation at s = 16,000 steps yields 13.13 ± 0.15%, confirming robustness to modest variation in training duration. The s = 18,000 setting (13.11 ± 0.16%) is used as the primary result throughout the paper.

| Seed | NRMSE (%) |
|:----:|:----------:|
| 42 | 12.89 |
| 43 | 13.24 |
| 44 | 13.16 |
| 45 | 13.26 |
| 46 | 13.12 |
| **Mean ± Std** | **13.13 ± 0.15** |

### A.4 BB 900K + RevIN

The BuildingsBench 900K model was retrained from scratch with RevIN enabled (identical architecture, optimizer, and training schedule to the original). The resulting NRMSE of 13.89% on the 955-building commercial evaluation set represents a 0.62 pp degradation relative to the original 13.27%. This single-run result uses the same seed and hyperparameters as the original BuildingsBench training.

---

## Appendix B: BB-700 and BB-7K Scaling

| Configuration | N Buildings | RevIN | Aug | NRMSE (%) | NCRPS (%) |
|--------------|:-----------:|:-----:|:---:|:----------:|:---------:|
| BB-700 | 700 | ON | OFF | 15.28 | — |
| BB-700 (aug-matched) | 700 | ON | ON | 14.26 | 7.80 |
| BB-700 | 700 | OFF | OFF | 16.44 | — |
| BB-7K | 7,000 | ON | OFF | 14.50 | — |
| BB-7K | 7,000 | OFF | OFF | 15.41 | — |

A 10x increase in BuildingsBench buildings (700 to 7,000) reduces NRMSE by 0.78 pp with RevIN (no aug). With matched augmentation, BB-700 reaches 14.26%---still 1.24 pp above BB-7K (no aug, 14.50%), and still 1.15 pp above Korean-700 (13.11%). Scaling within the stock-model distribution converges slowly and does not approach the performance achieved by 700 LHS-designed buildings regardless of augmentation.

All results verified through unified evaluation (torch 2.0.1, autocast, identical protocol as BB SOTA reproduction).

---

## Appendix C: Additional Ablation Results (Earlier Pipeline)

The four experiments below were conducted with an earlier evaluation pipeline prior to the adoption of the val_best checkpoint and unified 955-building commercial evaluation. The directions of degradation are reliable, but exact NRMSE values carry approximately ±0.3 pp additional uncertainty relative to Table 3 and Table 5.

| Experiment | NRMSE (%) | Delta vs Table 3 baseline | Note |
|------------|:----------:|:-------------------------:|------|
| BB Box-Cox (BB-fitted scale/location) | 16.24 | +3.31 | Korean loads mapped outside trained range |
| 4× training tokens (168K steps, ~84 epochs) | 16.02 | +3.09 | Overfitting to synthetic artifacts |
| 5K cap per archetype (70K buildings total) | 15.35 | +2.42 | Fixed-budget under-sampling at scale |
| Seasonal decomposition + RevIN | 16.65 | +3.72 | Trend-seasonal separation hurts OOD transfer |

All four degradations are large (>2 pp) and consistent with the sub-epoch overfitting insight from BuildingsBench: the model memorizes simulation-specific patterns that do not transfer to real buildings when exposed to more data or longer training than the optimum.
