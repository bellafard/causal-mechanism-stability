# Does behavior stabilize before mechanism?

A small causal interpretability study of learning dynamics in a transformer.

## Research question

When a model reaches reliable task performance, has its internal causal mechanism
also stabilized?

This project trains a two layer transformer on the complete modular addition truth
table and saves dense checkpoints throughout training. At every checkpoint it
measures:

1. **Behavior**, using accuracy across the complete truth table.
2. **Representation**, using centered kernel alignment with the final checkpoint.
3. **Causal mechanism**, using the loss increase caused by ablating each attention
   head and groups of MLP neurons, then comparing that causal importance profile
   with the final model.

The key comparison is the first checkpoint after which each measure never falls
below its prespecified threshold. The experiment therefore tests, rather than
assumes, whether exhaustive behavioral mastery precedes internal stabilization.

## Why this is interpretability

Attention head and grouped MLP ablations are interventions. They ask which
components are necessary for the trained computation, not merely which activations
correlate with the answer. Tracking these interventions through learning
distinguishes a stable causal profile from a transient correlate.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiment.py
```

A smoke test is available:

```bash
.venv/bin/python experiment.py --quick --output quick_results
```

The complete run writes raw checkpoint metrics, head ablation results, a summary,
and publication ready figures to `results/`.

Figures can be regenerated without retraining:

```bash
.venv/bin/python experiment.py --plot-only --output results
```

## Prespecified criteria

For the checkpoint and every checkpoint that follows:

- Behavioral mastery: accuracy across the truth table at least 0.99.
- Representational stability: mean CKA to the final checkpoint at least 0.95.
- Causal stability: correlation of the head ablation profile with the final
  checkpoint at least 0.90.

These thresholds are declared before examining the result to reduce narrative
flexibility. Results should be interpreted across seeds, and a failure to find
separation is informative.

## Limitations

- Modular addition is a controlled algorithmic task, not language modeling.
- Because training covers the complete truth table, the study addresses exhaustive
  task mastery rather than generalization.
- Component ablation is coarse and can miss distributed or redundant mechanisms.
- Similar causal importance profiles do not prove identical algorithms.
- The final checkpoint is a practical reference, not ground truth.

The natural next step is activation patching on specific token positions, followed
by analysis of whether the same causal subspace persists across seeds.

## Result

Across three seeds, accuracy on the complete truth table and the causal
intervention profile were stable by step 100, the first checkpoint after
initialization. Representational similarity reached its prespecified stability
criterion later, at step 400 in one seed and step 500 in two seeds.

The result does not support the strongest initial hypothesis that component level
causal importance continues reorganizing after behavioral mastery. Instead, it
reveals a distinction between levels of internal organization: the allocation of
causal importance across attention heads and MLP groups stabilizes early, while
the geometry of the residual representation continues to consolidate.

![Stability timeline](results/stability_timeline.png)

See [RESULTS.md](RESULTS.md) for the concise report.
