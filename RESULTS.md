# Behavioral mastery, grokking, and internal stability

## Summary

A two layer transformer learned addition modulo 17 under two data regimes. In the full-table control, all 289 pairs were used for both optimization and evaluation. Across three random seeds, `full_table_accuracy` and the causal importance profile stabilized by step 100, while residual representation geometry stabilized between steps 400 and 500.

In a separate 50% training split, training accuracy stabilized at step 500 but held-out accuracy did not remain above 95% until step 33,500. The causal intervention profile stabilized at step 14,000, before held-out behavior, while representation geometry continued changing through the 50,000-step endpoint.

Across these runs, the three measures crossed their stability thresholds at
different times. The ordering in the full-table control repeated across three
seeds; the ordering in the held-out condition remains exploratory until it is
replicated across seeds.

## Relation to grokking

[Power et al. (2022)](https://arxiv.org/abs/2201.02177) defined grokking as generalization long after a model has overfit its training set. [Nanda et al. (2023)](https://arxiv.org/abs/2301.05217) reverse engineered modular addition and separated training into memorization, circuit formation, and cleanup.

The full-table condition here intentionally removes the training and held-out gap, so it is a non-grokking control. The held-out condition restores that gap and uses stronger weight decay. A 40% training pilot failed to grok within 70,000 steps; a 50% split produced the reported phase gap.

## Design

Both conditions used a two layer transformer with four attention heads per layer, an embedding width of 64, and an MLP width of 128.

The full-table control used all 289 pairs, learning rate 0.003, weight decay 0.01, 5,000 steps, checkpoints every 100 steps, and three seeds.

The held-out condition used a 50/50 train and held-out split, learning rate 0.0003, weight decay 1.0, 50,000 steps, checkpoints every 500 steps, and one seed.

Three stability quantities were used:

1. Full-table mastery, `full_table_accuracy` of at least 0.99, or held-out generalization, `heldout_accuracy` of at least 0.95.
2. Representational stability, mean centered kernel alignment of at least 0.95 with the final checkpoint.
3. Causal stability, correlation of at least 0.90 between the checkpoint and final causal intervention profiles.

A criterion was considered stable at the first checkpoint after which it never fell below its threshold.

The control criteria were declared before its final three-seed run. The 0.95
held-out criterion was selected after a pilot and should be treated as
exploratory.

The causal profile contained the loss increase caused by ablating each attention head and each of eight equally sized MLP neuron groups per layer. This gives 24 interventional measurements at every checkpoint.

## Full-table control

| Seed | Full-table stability | Causal stability | Representational stability | Final full-table accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100 | 400 | 1.00 |
| 1 | 100 | 100 | 500 | 1.00 |
| 2 | 100 | 100 | 500 | 1.00 |

![Stability timeline](results/stability_timeline.png)

The dashed lines are the stability criteria. Curves show the mean across seeds,
and shaded regions show one standard deviation.

## Held-out grokking condition

| Seed | Training stability | Causal stability | Held-out stability | Representational stability | Final held-out accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 500 | 14,000 | 33,500 | 50,000 | 0.979 |

![Grokking stability timeline](grokking_50pct_results/stability_timeline.png)

The dotted blue curve is training accuracy. The orange curve is accuracy on held-out pairs. CKA and ablation profiles are also evaluated only on held-out pairs in this condition.

## Interpretation

In the control, the model's representation continued to approach its final geometry for several hundred steps after it answered every possible input correctly. Yet the relative causal importance assigned to broad components was already highly correlated with the final model.

In the held-out run, the causal profile came to resemble that of the final model
before delayed generalization, while the residual representation continued
changing afterward. This provides a concrete hypothesis for replication: coarse
causal allocation may become stable before held-out behavior even while
representation geometry is still changing.

## Limitations

The intervention is deliberately simple. Stable component importance does not
prove that the implemented algorithm is unchanged. Ablation can miss redundancy,
interactions among components, and changes within an MLP group. CKA also measures
representation geometry rather than semantic identity. The grokking condition
has not yet been replicated across seeds, and the 50% split was selected after a
40% pilot failed to generalize within the training budget.

A natural follow up would use activation patching to localize information by position and layer, then test whether the causal subspace itself remains stable across checkpoints and seeds.
