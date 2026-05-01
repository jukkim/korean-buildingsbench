# Results Registry — Korean BuildingsBench

모든 논문 보고 수치의 출처·재현 방법·검증 로그를 기록한다.
이 파일이 "이 숫자가 어디서 왔는가?"의 유일한 답이다.

---

## 공통 조건

| 항목 | 값 |
|------|------|
| 평가 프로토콜 | BB 955 commercial (BDG-2:611 + Electricity:344), NRMSE median |
| 평가 환경 | bb_repro conda (torch 2.0.1) 또는 Python 3.13 + torch 2.9.1 |
| 체크포인트 선택 | `_best.pt` (val_loss 기준, Korean val split) |
| 학습 | `--bb_eval_interval 0` (inline BB eval 없음) |
| Box-Cox | lambda=-0.067, `data/korean_bb/metadata/transforms/boxcox.pkl` |
| 평가 커맨드 | `python scripts/evaluate_bb.py --checkpoint <ckpt> --config <cfg> --commercial_only` |

---

## Table 3: Main Comparison

| Model | NRMSE | NCRPS | Seeds | Checkpoint | Config | 재현 로그 |
|-------|:---:|:---:|:---:|----------|--------|---------|
| BB SOTA-M | 13.27% | —† | 공식 | `external/BuildingsBench_data/checkpoints/Transformer_Gaussian_M.pt` | — | `logs/bb_sota_repro_M.log` |
| BB 900K+RevIN | 13.89% | 7.76% | 1 | `bb900k_revin_step590000.pt` | M-v3-3k.toml | 5090: `logs/ncrps_bb900k_revin.log` |
| Korean-700 ON | 13.11±0.17% | 7.14±0.03%‡ | 42-46 | `*_s18000_revin_on_seed{42-46}_best.pt` | M-v3-3k.toml | 5090: `logs/ncrps_on_seed{42-46}.log` |
| Korean-700 OFF | 14.72±0.28% | 8.29%§ | 42-44 | `*_s18000_revin_off_seed{42-44}_best.pt` (42,43 retrained) | M-v3-3k-norevin.toml | 5090: `logs/ncrps_off_seed{42,44}.log` |
| BB-700 (aug-matched) ON | 14.26% | 7.80% | 42 | `*_bb700_aug_s18000_seed42_best.pt` | M-v3-3k-bb.toml | 5090: `logs/ncrps_bb700_aug_seed42.log` |
| BB-700 (no-aug) ON | 15.28% | — | 42 | `*_bb700_s18000_revin_on_best.pt` | M-v3-3k.toml | `archive/logs/bb700_evaluate_bb.log` |
| BB-700 OFF | 16.44% | — | 42 | `*_bb700_s18000_revin_off_best.pt` | M-v3-3k-norevin.toml | `archive/logs/bb700_off_evaluate_bb.log` |
| Persistence Ensemble | 16.68% | — | — | — | — | **출처: BB 원 논문 Table 5** (Emami et al., NeurIPS 2023). 자체 실험값 아님. |

†BB SOTA-M NCRPS는 BC protocol 불일치로 보고 불가. 우리 BC로 평가 시 CVRMSE=15.05%, NCRPS=8.32% (5090: `logs/ncrps_bb_sota.log`).
‡4-seed mean (42,43,44,46). seed45는 retrained checkpoint NCRPS 미평가.
§2-seed mean (42,44). seed43은 broken checkpoint NCRPS 미평가.

### NCRPS 평가 조건
- 평가 스크립트: `scripts/evaluate_bb.py` (NCRPS 지원 버전, 2026-04-29)
- Box-Cox: Korean lambda=-0.067, `data/korean_bb/metadata/transforms/boxcox.pkl`
- Delta method: σ_orig ≈ scale_ × |λμ_unstd+1|^(1/λ-1) × σ_bc
- 정규화: CRPS / |actual_kWh|, building별 median 집계

### No-aug Ablation (Korean-700, seed 42-44)

| Seed | NRMSE | NCRPS | Checkpoint | 로그 |
|:---:|:---:|:---:|----------|------|
| 42 | 13.48% | 7.95% | `*_korean700_noaug_s18000_seed42_best.pt` | 5090: `logs/ncrps_noaug_seed42.log` |
| 43 | 13.89% | 8.21% | `*_korean700_noaug_s18000_seed43_best.pt` | 5090: `logs/ncrps_noaug_seed43.log` |
| 44 | 13.65% | 8.31% | `*_korean700_noaug_s18000_seed44_best.pt` | 5090: `logs/ncrps_noaug_seed44.log` |
| **Mean** | **13.67%** | **8.16%** | — | — |

→ aug 효과: NRMSE -0.56pp (13.67→13.11%), NCRPS -1.02pp (8.16→7.14%)

### Korean-700 ON 개별 seed (Appendix A.1)

| Seed | NRMSE | NCRPS | Checkpoint | 비고 |
|:---:|:---:|:---:|----------|------|
| 42 | 12.93% | 7.10% | `*_s18000_revin_on_seed42_best.pt` | 정상 (epoch 9) |
| 43 | 13.06% | 7.16% | `*_s18000_revin_on_seed43_best.pt` | 정상 (epoch 9) |
| 44 | 13.10% | 7.16% | `*_s18000_revin_on_seed44_best.pt` | 정상 (epoch 9) |
| 45 | 13.39% | n.e. | `*_retrain_on_45_best.pt` | 재학습 (원본 epoch 2에서 중단); NCRPS는 retrained ckpt 미평가 |
| 46 | 13.07% | 7.14% | `*_s18000_revin_on_seed46_best.pt` | 정상 (epoch 9) |

### Korean-700 OFF 개별 seed (Appendix A.2)

| Seed | NRMSE | NCRPS | Checkpoint | 비고 |
|:---:|:---:|:---:|----------|------|
| 42 | 14.81% | 8.17% | `*_retrain_off_42_best.pt` | 재학습 (원본 epoch 6에서 중단) |
| 43 | 14.94% | n.e. | `*_retrain_off_43_best.pt` | 재학습 (원본 epoch 0에서 중단); NCRPS는 broken ckpt 미평가 |
| 44 | 14.40% | 8.40% | `*_s18000_revin_off_seed44_best.pt` | 정상 (epoch 9) |

---

## Table 4: N-Scaling

모든 값: seed=42, s=18000, bb_eval_interval=0, `_best.pt` 평가.
n=5만 5-seed, n=50은 Table 3 seed42와 동일.

| n | Buildings | NRMSE | 실행 머신 | 로그 |
|---|:---:|:---:|:---:|------|
| 1 | 14 | 14.72% | 4090 | `logs/n1_n5_valbest_main.log` |
| 2 | 28 | 14.08% | 5090 | 5090: `nscale_valbest_main.log` |
| 3 | 42 | 13.47% | 5090 | 〃 |
| 4 | 56 | 13.45% | 5090 | 〃 |
| 5 | 70 | 13.28±0.12% (5-seed) | 4090 | `logs/n5_multiseed_main.log` |
| 6 | 84 | 13.33% | 4090 | `logs/n6789_valbest_main.log` |
| 7 | 98 | 13.21% | 4090 | 〃 |
| 8 | 112 | 13.27% | 4090 | 〃 |
| 9 | 126 | 13.26% | 4090 | 〃 |
| 10 | 140 | 13.18% | 5090 | 5090: `nscale_valbest_main.log` |
| 20 | 280 | 13.23% | 5090 | 〃 |
| 30 | 420 | 13.13% | 5090 | 〃 |
| 40 | 560 | 13.23% | 5090 | 〃 |
| 50 | 700 | 12.93% | 4090 | `logs/valbest_all.log` (seed42) |
| 60 | 840 | 13.18% | 5090 | 5090: `nscale_valbest_main.log` |
| 70 | 980 | 13.20% | 5090 | 〃 |
| 80 | 1120 | 13.15% | 5090 | 〃 |

### n=5 개별 seed

| Seed | NRMSE | Checkpoint |
|:---:|:---:|----------|
| 42 | 13.24% | `*_nscale_n5_s18000_valbest_best.pt` |
| 43 | 13.27% | `*_n5_s18000_seed43_best.pt` |
| 44 | 13.22% | `*_n5_s18000_seed44_best.pt` |
| 45 | 13.50% | `*_n5_s18000_seed45_best.pt` |
| 46 | 13.19% | `*_n5_s18000_seed46_best.pt` |

### 재현 커맨드 (n-scaling 단일 포인트)

```bash
python scripts/train.py \
  --config configs/model/TransformerWithGaussian-M-v3-3k.toml \
  --train_index train_weekly_{N}.csv \
  --val_index val_weekly_{N}.csv \
  --max_steps 18000 --warmup_steps 500 \
  --note nscale_n{N}_s18000_valbest \
  --augment --bb_eval_interval 0 --num_workers 0 --seed 42

python scripts/evaluate_bb.py \
  --checkpoint checkpoints/TransformerWithGaussian-M-v3-3k_nscale_n{N}_s18000_valbest_best.pt \
  --config configs/model/TransformerWithGaussian-M-v3-3k.toml \
  --commercial_only
```

---

## Table 5: Ablation (seed=42 단일)

| Experiment | NRMSE | Delta | 비고 |
|------------|:---:|:---:|------|
| Baseline (RevIN ON) | 12.93% | — | = Table 3 seed42 |
| RevIN OFF | 14.72% | +1.79 | 3-seed mean (Table 3) |
| BB Box-Cox | 16.24% | +3.31 | 이전 실험, inline eval |
| 4x tokens | 16.02% | +3.09 | 이전 실험, inline eval |
| 5K cap | 15.35% | +2.42 | 이전 실험, inline eval |
| Seasonal decomp | 16.65% | +3.72 | 이전 실험, inline eval |

> 주의: BB Box-Cox ~ Seasonal decomp은 이전 실험(bb_best)의 inline eval 값.
> baseline이 12.93%(val_best)으로 변경되었으므로 delta는 근사치.
> 방향성(큰 악화)은 확실하므로 논문 보고에 문제 없음.

---

## Table 6: Convenience Stores

| Model | All 218 | 100-store subset (2022, 289days) | 120-store subset (2024-25, 1yr) | 로그 |
|-------|:---:|:---:|:---:|------|
| Korean-700 | 12.30% | 17.42% | 10.22% | `logs/korean_stores_valbest.log` |
| BB SOTA-M | 13.14% | — | — | 〃 |

체크포인트: `*_s18000_revin_on_seed42_best.pt` (val_best)

> 100-store(17.42%) vs 120-store(10.22%) 편차는 데이터 품질 차이: 2022년 100개는 289일 부분 기간+갭 존재, 2024-25년 120개는 완전한 1년 데이터.

---

## Steps Sweep (Supplementary, n=50 seed=42)

| steps | NRMSE | 머신 | 로그 |
|:---:|:---:|:---:|------|
| 10000 | 13.49% | 5090 | 5090: `n234_steps_sweep_main.log` |
| 11000 | 13.32% | 5090 | 〃 |
| 12000 | 13.28% | 5090 | 〃 |
| 13000 | 13.29% | 5090 | 〃 |
| 14000 | 13.23% | 5090 | 〃 |
| 15000 | 13.20% | 5090 | 〃 |
| 16000 | 12.91% | 4090 | `logs/s16000_valbest_eval.log` |
| 17000 | 13.17% | 5090 | 5090: `n234_steps_sweep_main.log` |
| 18000 | 12.93% | 4090 | `logs/valbest_all.log` |
| 19000 | 13.20% | 5090 | 5090: `n234_steps_sweep_main.log` |
| 20000 | 13.17% | 5090 | 〃 |

---

## 5090 로그 위치

5090 결과 로그는 `C:\Korean_BB\logs\` (192.168.1.23)에 있음:
- `nscale_valbest_main.log` — n=10~80 결과
- `n234_steps_sweep_main.log` — n=2,3,4 + steps sweep 결과

---

---

## BB-700 aug-matched 체크포인트 (2026-04-29 추가)

| 파일 | NRMSE | NCRPS | 위치 |
|------|:---:|:---:|------|
| `TransformerWithGaussian-M-v3-3k-bb_bb700_aug_s18000_seed42_best.pt` | 14.26% | 7.80% | 5090: `C:\Korean_BB\checkpoints\` |

학습 조건: `--augment --bb_eval_interval 0 --seed 42 --max_steps 18000`, BB-700 index (stride=24, 245K windows), config `M-v3-3k-bb.toml` (RevIN ON)

*생성: 2026-04-23 | 마지막 수정: 2026-04-29 (NCRPS 평가 완료 + BB-700+aug 추가)*
