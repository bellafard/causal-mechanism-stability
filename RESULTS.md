# Behavioral mastery, causal stability, and representational consolidation

## Summary

A two layer transformer learned all 289 examples in the addition table modulo 17. Across three random seeds, exhaustive behavioral accuracy and the causal importance profile of model components stabilized by training step 100. The geometry of the residual representation stabilized later, between steps 400 and 500.

The initial hypothesis was that behavior would stabilize before the causal mechanism. The result did not support that claim at the resolution of the interventions used here. Instead, it showed that conclusions about internal stability depend on the level of description: coarse causal allocation was stable while representational geometry continued to change.

## Design

The model had two transformer layers, four attention heads per layer, an embedding width of 64, and an MLP width of 128. It was trained on the complete modular addition truth table for 5,000 steps. Checkpoints were evaluated every 100 steps across three random seeds.

Three quantities were prespecified:

1. Behavioral mastery, accuracy of at least 0.99 on the complete truth table.
2. Representational stability, mean centered kernel alignment of at least 0.95 with the final checkpoint.
3. Causal stability, correlation of at least 0.90 between the checkpoint and final causal intervention profiles.

A criterion was considered stable at the first checkpoint after which it never fell below its threshold.

The causal profile contained the loss increase caused by ablating each attention head and each of eight equally sized MLP neuron groups per layer. This gives 24 interventional measurements at every checkpoint.

## Result

| Seed | Behavioral stability | Causal stability | Representational stability | Final accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 100 | 400 | 1.00 |
| 1 | 100 | 100 | 500 | 1.00 |
| 2 | 100 | 100 | 500 | 1.00 |

![Stability timeline](results/stability_timeline.png)

The dashed lines are the criteria declared before the final run. Curves show the mean across seeds, and shaded regions show one standard deviation.

## Interpretation

The model's representation continued to approach its final geometry for several hundred steps after it answered every possible input correctly. Yet the relative causal importance assigned to broad components was already highly correlated with the final model.

This suggests a useful distinction. A system can continue consolidating its representation without reallocating causal responsibility at the granularity of attention heads and MLP groups. Behavioral, representational, and causal stability should therefore be measured separately rather than treated as synonyms.

<!-- ## What this does not establish -->

The intervention is deliberately simple. Stable component importance does not prove that the implemented algorithm is unchanged. Ablation can miss redundancy, interactions among components, and changes within an MLP group. CKA also measures representation geometry rather than semantic identity. Finally, modular addition is a complete, controlled task and does not test generalization in a language model.

A natural follow up would use activation patching to localize information by position and layer, then test whether the causal subspace itself remains stable across checkpoints and seeds.
