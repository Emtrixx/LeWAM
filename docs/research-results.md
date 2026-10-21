# PushT research results — 21 October 2026

The five-model study met the agreed local stopping targets of **71% policy** and
**70% DS-CEM mean maximum raw coverage**. This is an independent implementation;
exact equivalence to the paper's plotted coverage aggregation remains unresolved.

| Frozen selected method | Maximum raw coverage, mean ± model SD | Terminal raw coverage | Strict >95% successes |
| --- | ---: | ---: | ---: |
| Policy, original BatchNorm | **73.21% ± 3.80 pp** | 69.48% | 105/250 (42.0%) |
| DS-CEM, train-only BN calibration | **73.53% ± 4.20 pp** | 69.71% | 116/250 (46.4%) |

Models 1–5 each trained for 300 epochs / 47,400 updates, with 300 full validation
passes, 224px inputs, batch 128, BF16 and all four objectives. Every inference arm
used the same 50 starts (0–49), 300 primitive steps and the Lissajous condition.
Models 1–4 ran on separate A100s, model 5 on H100. Interrupted A100 jobs resumed
the complete saved state and original training seeds; they are not extra models.

The training source was `7a60273a56f5c413f7f5c7ef49d0e35aed2b79a1`, and evaluation
used `c074699ebe32f74c53cb1587d532ab6761069a6b`. Both are retained in Git history.
The [machine-readable record](results/pusht-20261021.json) includes all four arms,
per-model means, evaluation/checkpoint hashes and the prospectively fixed selectors.
Raw training checkpoints and videos remain in the research archives.

## Calibration was not a consistent improvement

| Inference arm, original five-model cohort | Maximum raw coverage | Terminal raw coverage | Binary success |
| --- | ---: | ---: | ---: |
| Policy, original BN | 73.21% | 69.48% | 42.0% |
| Policy, calibrated BN | 72.46% | 67.74% | 40.0% |
| DS-CEM, original BN | 74.86% | 71.38% | 42.8% |
| DS-CEM, calibrated BN | 73.53% | 69.71% | 46.4% |

Calibration changed paired mean maximum coverage by **−0.75 pp policy** and
**−1.33 pp DS-CEM**. Its strong result on one development checkpoint did not
establish a general benefit. Calibration therefore stays **disabled by default**;
the table preserves the original selector rather than selecting winners afterward.

A separate BN-both study prospectively selected models **1, 2, 3, 4, 6**, excluding
models 0 and 5 as development evidence. It completed at **71.20% policy / 72.67%
DS-CEM** maximum raw coverage, terminal coverage 66.53% / 69.44%, and binary success
37.2% / 43.2%. Four models overlap the first study, so these are not independent
five-model replications. Paired calibration effects were again negative on average
(−3.24 pp policy / −2.05 pp DS-CEM).

## Interpretation and verification

Five model means provide the training replicates; the repeated 50 starts do not
create 250 independent trained models. Descriptive 95% t intervals for the original
selected policy and DS-CEM means are 68.49–77.93% and 68.31–78.74%. The stopping
rule used the observed means, not confidence lower bounds. Coverage, terminal
coverage, normalized environment reward and binary success are different metrics.

The paper's Figure 6 plots approximately 73% policy / 72% DS-CEM PushT coverage.
The local results are numerically close, but raw versus normalized and maximum
versus terminal aggregation are not clearly pinned by the paper. Goal retrieval,
reset context and distractor rendering also include independent implementation
assumptions. This is not a claim of exact paper reproduction or superiority to
historical LeWM models evaluated with different action, goal and horizon contracts.

Independent audits verified all six complete checkpoints, saved-state lineage,
source/runtime/data identity, full validation, four-arm evaluations, **1,200 playable
videos**, durable publication and once-only allocation closeouts. One original
final publication failed after successful science; a separately audited recovery
preserved its results and original failure. Further PushT improvement seeds and
the deferred BC-only experiment were stopped after the target was met.

See [reproduction assumptions](reproduction.md), [initial validation](validation.md)
and [the reference paper](https://arxiv.org/html/2610.12407v1).
