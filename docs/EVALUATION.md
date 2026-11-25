# Evaluation status

## Smoke-run verification — 2026-11-24

The official Sen1Floods11 `S1Hand`/`LabelHand` data was downloaded and its published split membership was verified locally:

| Split | Verified chips |
| --- | ---: |
| Train | 252 |
| Validation | 89 |
| Test | 90 |
| Bolivia geographic hold-out | 15 |

One **CPU smoke run** was executed using only 4 train chips, 2 validation chips, one epoch, and an 8-channel U-Net. It exists solely to verify data loading, ignore-label handling, loss, checkpointing, MC-dropout evaluation, and ONNX export. It is not a training experiment and must never be deployed.

| Split | Learned smoke IoU | Deterministic baseline proxy IoU | Otsu VH IoU |
| --- | ---: | ---: | ---: |
| Official test | 0.1251 | 0.4355 | 0.2445 |
| Bolivia hold-out | 0.1586 | 0.6288 | 0.3655 |

The smoke checkpoint underperformed the baseline by a substantial margin, so it is not eligible for backend or frontend use. The full training/evaluation commands remain the only path to promotion.

The existing deterministic flood implementation remains a baseline only. Its output is not an ML-model evaluation result.
