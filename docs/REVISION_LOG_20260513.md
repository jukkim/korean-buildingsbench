# Revision Log — 2026-05-13 GPT 4-Round Review

Post-submission refinements applied to `paper/main.tex` based on 4 rounds of GPT review.
Building Simulation (BUIL-S-26-00752) desk reject 후 Energy and AI 투고 준비용.

---

## Round 1: Structural & Factual

| # | Issue | Fix |
|---|-------|-----|
| 1 | Feature-space Table 7 missing | Added Table 7 with 8D coverage metrics |
| 2 | "Proposition 1" overly formal | Renamed to "Heuristic relation (informal)", removed QED box, added caveat |

## Round 2: Factual Consistency (9 items)

| # | Issue | Fix |
|---|-------|-----|
| 1 | 969 vs 955 building count | Table 7 caption → "955 real evaluation buildings" |
| 2 | "14 of 16 DOE" factually wrong | Rewritten: 12 direct + 2 derived + 4 excluded, all 16 accounted |
| 3 | Feature coverage contradicts claim | Section 5.4 title → "Feature-space proximity does not explain the gain" |
| 4 | BB-700 single subset caveat | Added caveat in Section 5.1, Limitations, Conclusion |
| 5 | NCRPS vs RPS confusion | Added evaluation protocol note: different scripts, not directly comparable |
| 6 | Appendix C overstatement | Reframed as "Exploratory Ablation Results (Earlier Pipeline)" |
| 7 | Data availability vague | Added IDF templates, peer review provision |
| 8 | "bound" → "decomposition" | Fixed in Heuristic relation body |
| 9 | Operating hours concentration | Added explanation: high baseload and nighttime equipment fractions |

## Round 3: Formatting & Precision (10 items)

| # | Issue | Fix |
|---|-------|-----|
| 1 | Appendix tables float before headers | `\usepackage{float}` + `[H]` + `\FloatBarrier` |
| 2 | "Appendix Appendix C" duplication | Changed to `Appendix~C` direct reference |
| 3 | Secondary school + small hotel unaccounted | Listed all 4 excluded DOE types |
| 4 | Highlight #3 missing caveat | Updated to "LHS shows 1.33 pp gain over one equal-size stock-model control" |
| 5 | "full operational schedule space" | → "designed operational parameter space" |
| 6 | Figure 4 delta discrepancy (1.88 vs 1.61) | Added caption note: seed-42 vs five-seed mean |
| 7 | CI direction ambiguous | Specified "NRMSE(BB-900K) - NRMSE(K-700)" |
| 8 | Figure 1 label | "14 DOE Archetypes" → "14 DOE-Derived Archetypes" |
| 9 | NCRPS handling confirmed | BB-900K NCRPS cell left blank with footnote |
| 10 | Data availability strengthened | "will be provided for peer review" |

## Round 4: Final Confirmation

| # | Issue | Fix |
|---|-------|-----|
| 1 | Conclusion missing single-subset caveat | Added "(1) in the single-subset equal-scale control" |

GPT declared paper "submission-ready" after Round 4.

## Equation Assessment

Reviewed whether more equations are needed. Conclusion: current set (Eq 1-5) is sufficient.
- Eq 1: NRMSE
- Eq 2-3: Gaussian NLL / probabilistic loss
- Eq 4-5: Heuristic relation (covering-radius decomposition)
Adding RevIN equations (2 lines) would be optional but not required for this journal.

---

## Files Modified

| File | Changes |
|------|---------|
| `paper/main.tex` | ~30 edits across 4 review rounds |
| `paper/main.pdf` | Recompiled (29 pages, 552 KB) |
| `paper/main.docx` | Regenerated via pandoc |
| `paper/generate_figures.py` | Fig 1 box label updated |
| `paper/figures/*.pdf,*.png` | All 4 figures regenerated |
| `docs/paper_final.md` | SSOT synced with all LaTeX changes |
| `docs/highlights_energy_ai.md` | Highlight #3 updated |
| `docs/PAPER_PLAN.md` | Journal status updated |
