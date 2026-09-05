# -*- coding: utf-8 -*-
"""Korean_BB 응답에 **계보 3축**을 싣는다 (2026-09-05 신설).

## 왜

응답에 `model.training_data = "Korean_BB_700_sims_revin_on"` 은 이미 있었다.
그런데 소비자(게이트웨이 봉투)는 그 문자열을 **해석할 수 없다** — 이게 시뮬인지
실측인지, 답이 관측인지 추론인지 판별하려면 소비자가 이 모델을 알아야 한다.

⛔ 그래서 게이트웨이가 상수 표로 대신 채우고 있었고, 그러면 세는 쪽이
*"자기가 넣은 걸 자기가 센다"* — 표에 줄을 더해도 백엔드는 안 바뀐다.

**계보는 답을 만든 쪽이 안다.** 여기서 3축으로 못 박는다.

⚠ 이미 있는 `model` 블록은 **건드리지 않는다** — 소비자가 그걸 읽고 있을 수 있다.
   `provenance` 를 **더할** 뿐이다.

## 축 셋 (게이트웨이 `slot_resolver.ResolvedSlot` 과 같은 정의)

    source_kind        meter | administrative | simulation | archetype | fixture
    inference_method   direct_map | derived | probabilistic_model | default
    assertion_status   source_reported | inferred | assumed | unknown
"""
from __future__ import annotations

_FORECAST = {
    "source_kind": "simulation",
    "inference_method": "probabilistic_model",
    "assertion_status": "inferred",
    "basis": ("Korean_BB TwGauss-M — **Korean_BB 700건 시뮬**(15채널)로 학습한 "
              "transformer_with_gaussian. 응답의 `model.training_data` = "
              "`Korean_BB_700_sims_revin_on` 이 같은 사실을 말한다"),
    "note": ("⚠ CVRMSE 12.93% 는 **LOCKED 판정값**이다(8.simulation/CLAUDE.md). "
             "다른 조건에서 잰 값과 섞지 마라.\n"
             "⛔ 시계열 3종은 이 모델 책임이고 LLM 은 **호출·서술만** 한다 — "
             "같은 문서 「시계열 책임 분담 SSOT」."),
}


def attach(result: dict, kind: str = "forecast") -> dict:
    """`result` 에 `provenance` 를 얹는다.

    ⚠ 이미 있으면 건드리지 않는다. dict 가 아니면 그대로 돌려준다.
    ⚠ 판본은 **응답의 `model` 블록에서** 읽는다 — 여기 박으면 즉시 stale 된다.
    """
    if not isinstance(result, dict) or result.get("provenance") or kind != "forecast":
        return result
    out = dict(_FORECAST)
    m = result.get("model")
    #: 계보는 아는데 **판본을 모르는 것**은 다른 결손이라 `null` 로 구별해 적는다.
    out["runtime"] = ({k: m[k] for k in ("id", "family", "training_data")
                       if isinstance(m, dict) and m.get(k)} or None)
    result["provenance"] = out
    return result
