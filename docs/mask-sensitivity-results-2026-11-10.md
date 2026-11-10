# Outcome-mask sensitivity: results (2026-11-10)

Plan: `review-2026-11-10-fixes-plan.md`, fix 2. Report: `data/reports/research/audit-masks-20261110T165740.md`
(`uv run quant audit masks`; commit `762bb86`, feature build `20261110T163531`). Research period only;
nothing logged, no registry change.

## Question

The external review of 2026-11-10 noted that the research sample mixes the point-in-time universe with
three masks that use information after t:
- `quality`: a data-error price jump inside the target window;
- `execution`: the next open is at the ceiling;
- `period`: the target window crosses a period boundary.

Do the recorded research-period conclusions depend on them?

## Setup

- Six configurations:
  - `baseline`: no mask;
  - one configuration per mask;
  - `combined`: the recorded definition, used as the reference;
  - `combined_horizon`: all masks over each target's own window.
- Each configuration re-runs:
  - the 15 library patterns;
  - the decile scan (31 features);
  - the interaction scan (54 contrasts);
  - the event study (12 event types);
  - the cross-stock tests.
- Each configuration runs in a scratch results database. q-values are BH within that scratch family.
- Declared rule: a test is mask-sensitive when its effect changes sign, or its q (or p) crosses 0.05.
  It is **material** when it is significant under `combined` and its sign changes.
- Check: `combined` reproduces the recorded pattern lifts to within 0.00004. The small gap comes from new
  dividend adjustments since the original runs.

## Results

**Sample.** The masks remove 1.1% of point-in-time rows (10-session target):
- quality 0.02%;
- execution 0.41%;
- period 0.69%. Almost all of it falls in 2023 (7.5% of that year), where the 20-session window reaches into
  2024.

The 10-session mean excess moves from +0.090% (baseline) to +0.093% (combined).

**Concentration.**
- The execution mask is concentrated in momentum and order-flow patterns. It removes:
  - `imbalance_positive_price_down`: 5.2%;
  - `surge4_volume_close_high`: 2.7%;
  - `order_imbalance_d10`: 2.4%;
  - `momentum10_d10`: 1.6%.

  The reviewer's point is confirmed: the share is far above the 0.41% average in these patterns.
- The validated avoid signal `drop3_volume2_sellers` loses 0.37% of its hits to the execution mask and none
  to the other masks.

**Conclusions.** No material change in any analysis or configuration.

| Analysis | Tests | Mask-sensitive (any configuration) | Material | Rank agreement (min Spearman) |
|---|---|---|---|---|
| Patterns | 54 | 2 | 0 | 0.999 |
| Decile scan | 93 | 5 | 0 | 0.999 |
| Interactions | 54 | 3 | 0 | 0.998 |
| Event study | 36 | 0 | 0 | 0.996 |
| Cross tests | 22 | 1 | 0 | 0.999 |

What the flagged tests are:
- **Sign flips of noise.** Effects near zero with p or q of about 0.85 to 0.99, for example
  `imbalance_positive_price_down` at 5 sessions, or `close_vs_avg_price` IC at 20 sessions.
- **Borderline significance:**
  - `return_5d` and `excess_return_5d` IC at 10 sessions: p 0.040 → 0.058 without masks. The effect keeps
    its sign and size.
  - Industry momentum `group_momentum_63d` at 20 sessions: p 0.047 → 0.057. This hypothesis (P4b-G2) has
    already failed validation and is closed.
  - Interactions `reversal_1w x breadth` (q 0.060 → 0.047) and `volume_surge x foreign` (0.053 → 0.036).
    These move the other way: they become candidates without the masks.

**Interaction candidates.** `combined` gives 14. Every other configuration keeps all 14 and adds at most
two borderline ones. None is dropped.

## Reading

- The masks are not what makes the recorded research conclusions. Removing them never turns a significant
  effect around, and rankings agree to at least 0.996.
- Where the masks change a borderline result, they make it **less** significant as often as more:
  - `return_5d`, `excess_return_5d` and group momentum lose significance without the masks;
  - two interactions gain it.
- The direction of the execution mask is not uniformly conservative. It removes winners from the universe
  baseline, and it removes them disproportionately from momentum and order-flow patterns. Here the net
  effect was below the decision threshold.
- No registry state changes. The validated avoid signal and the failed or forward hypotheses are not
  re-decided: validation and holdout cannot be reused.
- New research definitions should use the point-in-time universe with explicitly chosen masks over the
  target's own window (`mask_window="horizon"`). The legacy 20-session window over-censors 5- and 10-session
  targets near period ends. In the research liquid universe, the period mask removes 6,662 rows, against
  1,670 for the 5-session window: four times as many as needed.
