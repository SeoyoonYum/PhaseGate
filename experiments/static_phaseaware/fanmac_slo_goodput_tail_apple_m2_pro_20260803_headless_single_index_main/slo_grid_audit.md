# SLO Grid Audit

- Timestamp: `2026-08-03T02:19:46.291235+02:00`
- Repository commit: `8ba6cbb3eb0d8e948f6ceef6898b753076022451`
- Source: `experiments/static_phaseaware/fanmac_main_apple_m2_pro_20260801/calibration_runs.csv`
- The grid was frozen before the new calibration and held-out evaluation.

| Policy | Valid runs | Normalized TPOT values | TPOT median/max | Normalized TTFT values | TTFT median/max |
|---|---:|---|---|---|---|
| fixed1 | 3 | 1.213883757, 1.196293539, 1.240876040 | 1.213883757 / 1.240876040 | 1.005985356, 1.002788907, 0.999569062 | 1.002788907 / 1.005985356 |
| phasegate4to1 | 3 | 1.225110906, 1.227952589, 1.210229021 | 1.225110906 / 1.227952589 | 1.014754223, 1.013317690, 1.009812339 | 1.013317690 / 1.014754223 |
| fixed2 | 3 | 1.391738233, 1.396119720, 1.388177939 | 1.391738233 / 1.396119720 | 1.012208153, 1.005575399, 1.004722832 | 1.005575399 / 1.012208153 |
| phasegate4to2 | 3 | 1.397356809, 1.399912952, 1.413621068 | 1.399912952 / 1.413621068 | 1.021702797, 1.011458695, 1.012590621 | 1.012590621 / 1.021702797 |

`relevant_max = 1.413621068172`

`B_max = 1.45`

Frozen grid: `{1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45}`

The extension is mechanical from old calibration extrema; no new held-out result was inspected or used to choose a favorable budget.
