# Power Model Fitting Results
_Generated: 2026-04-17 01:32:24_

## Fitted Model

```
P(f) = k3*f^3 + k2*f^2 + k1*f + k0

k3 = 2.676963e-07
k2 = -5.313477e-04
k1 = 4.448391e-01
k0 = 1.839749e+01
```

## Goodness of Fit

| Metric | Value |
|--------|-------|
| R^2    | 0.9845 |
| MAPE   | 3.78% |

## Raw Measurements

| SM Clock (MHz) | Mean Power (W) | Std (W) | Fitted P(f) (W) | Error (%) |
|---------------|---------------|---------|-----------------|----------|
|            210 |         90.99 |    1.47 |           90.86 |     0.15 |
|            270 |        102.80 |    1.77 |          105.04 |     2.17 |
|            345 |        125.72 |    1.93 |          119.62 |     4.86 |
|            420 |        135.69 |    1.57 |          131.33 |     3.21 |
|            480 |        140.91 |    1.26 |          139.10 |     1.29 |
|            555 |        137.76 |    1.74 |          147.38 |     6.98 |
|            630 |        140.60 |    3.08 |          154.69 |    10.02 |
|            690 |        154.18 |    3.67 |          160.30 |     3.97 |
|            765 |        174.59 |    3.78 |          167.59 |     4.01 |
|            840 |        183.65 |    5.22 |          175.81 |     4.27 |
|            915 |        193.61 |    4.23 |          185.64 |     4.12 |
|            975 |        201.80 |    2.46 |          195.12 |     3.31 |
|           1050 |        216.53 |    5.61 |          209.56 |     3.22 |
|           1125 |        225.38 |    5.78 |          227.51 |     0.94 |
|           1185 |        239.62 |    6.44 |          244.85 |     2.18 |
|           1260 |        254.32 |    8.22 |          270.82 |     6.49 |
|           1335 |        294.47 |    8.77 |          302.20 |     2.62 |
|           1410 |        354.43 |   10.77 |          339.66 |     4.17 |

## Notes

- Power measured via NVML `nvmlDeviceGetPowerUsage` (whole-card, mW resolution).
- Memory clock fixed at hardware default (1593 MHz for A800 SXM4 80GB; not software-controllable).
- `k0` is the polynomial intercept and does **not** equal `P_idle` (measured separately).
- Model valid only under saturated prefill load at the profiled prompt length (1024 tokens).
