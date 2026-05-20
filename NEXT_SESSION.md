# Korean_BB — 다음 세션 진입점

> 🔒 잠금 — Energy and AI 투고 준비 중 (APEN→E&B→BS desk reject 후). 활성 체크포인트 = `TransformerWithGaussian-M-v3-3k_bb700_s18000_revin_on_best.pt` (NRMSE 12.30% / CVRMSE 12.93%).

## 진입 작업 — 본선까지 잠금 유지

본 모델은 논문 투고 상태. **`frozen=true` (ai_model_registry.json)** — **추가 학습 / 가중치 변경 금지**.

### 가능 작업 (잠금 위반 X)

| 작업 | 잠금 영향 |
|------|:--:|
| API :8040 서빙 안정성 | 없음 — 가중치 변경 X |
| Gateway :8030 연동 검증 | 없음 |
| 평가 metric 추가 측정 (holdout 30d 등) | 없음 — 측정만, 학습 X |
| 본선 시연 시나리오 입력 검증 | 없음 |
| Type 1 분류 명시 (UNIFIED_GATEWAY_DESIGN §8.5) | 없음 — 메타데이터만 |

### 금지 작업 (잠금)

- ❌ 추가 학습 / fine-tune
- ❌ 가중치 변경
- ❌ Type 2 자동 재학습 트리거 적용 (Type 1 모델이므로)
- ❌ `frozen_for_demo` 에서 제거 (PRD §4.2 4 모델 필수)

### 잠금 해제 조건

- Energy and AI 논문 통과 또는 명시적 잠금 해제 결정
- 해제 후 v2 학습 — 별도 PRD 변경

## 활성 체크포인트 (ai_model_registry v1.1)

- ID: `korean_bb`
- backend_url: `http://localhost:8040`
- active_checkpoint: `TransformerWithGaussian-M-v3-3k_bb700_s18000_revin_on_best.pt`
- retrain_type: `type_1_sim`
- frozen: `true`

## SSOT

- `CLAUDE.md` (이 폴더 — FROZEN 명시)
- `FROZEN.md`
- `projects/energy-contracts/energy_contracts/schemas/ai_model_registry.json` v1.1

*작성: 2026-05-20*
