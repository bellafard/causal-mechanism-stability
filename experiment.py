#!/usr/bin/env python3
"""Measure whether behavior stabilizes before a transformer's causal mechanism."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    modulus: int = 17
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 128
    mlp_groups: int = 8
    steps: int = 5000
    checkpoint_every: int = 100
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    train_fraction: float = 1.0
    seeds: tuple[int, ...] = (0, 1, 2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, ablate_head: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, width = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), 1)
        weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        heads = weights @ v
        if ablate_head is not None:
            heads = heads.clone()
            heads[:, ablate_head] = 0
        merged = heads.transpose(1, 2).reshape(batch, seq, width)
        return self.out(merged), heads


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp_in = nn.Linear(config.d_model, config.d_mlp)
        self.mlp_out = nn.Linear(config.d_mlp, config.d_model)
        self.mlp_groups = config.mlp_groups

    def forward(
        self,
        x: torch.Tensor,
        ablate_head: int | None = None,
        ablate_mlp_group: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, heads = self.attn(self.ln1(x), ablate_head)
        x = x + attn_out
        hidden = F.gelu(self.mlp_in(self.ln2(x)))
        if ablate_mlp_group is not None:
            hidden = hidden.clone()
            group_size = hidden.shape[-1] // self.mlp_groups
            start = ablate_mlp_group * group_size
            end = hidden.shape[-1] if ablate_mlp_group == self.mlp_groups - 1 else start + group_size
            hidden[..., start:end] = 0
        x = x + self.mlp_out(hidden)
        return x, heads


class TinyTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.modulus + 1, config.d_model)
        self.position = nn.Parameter(torch.randn(3, config.d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.modulus, bias=False)

    def forward(
        self, tokens: torch.Tensor, ablate: tuple[str, int, int] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.token(tokens) + self.position
        activations: dict[str, torch.Tensor] = {"embedding": x}
        for layer, block in enumerate(self.blocks):
            head = (
                ablate[2]
                if ablate is not None and ablate[0] == "head" and ablate[1] == layer
                else None
            )
            mlp_group = (
                ablate[2]
                if ablate is not None and ablate[0] == "mlp" and ablate[1] == layer
                else None
            )
            x, heads = block(x, head, mlp_group)
            activations[f"layer_{layer}"] = x
            activations[f"heads_{layer}"] = heads
        return self.unembed(self.final_ln(x[:, -1])), activations


def make_data(config: Config, seed: int) -> tuple[torch.Tensor, ...]:
    pairs = [(a, b) for a in range(config.modulus) for b in range(config.modulus)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    split = int(len(pairs) * config.train_fraction)

    def encode(items: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor([[a, b, config.modulus] for a, b in items], dtype=torch.long)
        y = torch.tensor([(a + b) % config.modulus for a, b in items], dtype=torch.long)
        return x, y

    # Evaluation covers the complete truth table. This project asks when the
    # mechanism stabilizes after exhaustive behavioral mastery, not whether the
    # model generalizes beyond its training distribution.
    return *encode(pairs[:split]), *encode(pairs)


def accuracy_and_loss(
    model: TinyTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    ablate: tuple[str, int, int] | None = None,
) -> tuple[float, float]:
    with torch.no_grad():
        logits, _ = model(x, ablate)
        return (logits.argmax(-1) == y).float().mean().item(), F.cross_entropy(logits, y).item()


def centered_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    numerator = torch.linalg.norm(x.T @ y).square()
    denominator = torch.linalg.norm(x.T @ x) * torch.linalg.norm(y.T @ y)
    return (numerator / denominator.clamp_min(1e-12)).item()


def evaluate_checkpoint(
    model: TinyTransformer,
    state: dict[str, torch.Tensor],
    final_representations: list[torch.Tensor],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    model.load_state_dict(state)
    model.eval()
    train_acc, train_loss = accuracy_and_loss(model, train_x, train_y)
    test_acc, test_loss = accuracy_and_loss(model, test_x, test_y)
    with torch.no_grad():
        _, acts = model(test_x)
    ckas = [
        centered_cka(acts[f"layer_{layer}"][:, -1], final_representations[layer])
        for layer in range(model.config.n_layers)
    ]
    intervention_rows = []
    importance = []
    for layer in range(model.config.n_layers):
        for head in range(model.config.n_heads):
            ablated_acc, ablated_loss = accuracy_and_loss(
                model, test_x, test_y, ("head", layer, head)
            )
            delta_loss = ablated_loss - test_loss
            importance.append(delta_loss)
            intervention_rows.append(
                {
                    "component": "head",
                    "layer": layer,
                    "index": head,
                    "delta_loss": delta_loss,
                    "delta_accuracy": test_acc - ablated_acc,
                }
            )
        for group in range(model.config.mlp_groups):
            ablated_acc, ablated_loss = accuracy_and_loss(
                model, test_x, test_y, ("mlp", layer, group)
            )
            delta_loss = ablated_loss - test_loss
            importance.append(delta_loss)
            intervention_rows.append(
                {
                    "component": "mlp_group",
                    "layer": layer,
                    "index": group,
                    "delta_loss": delta_loss,
                    "delta_accuracy": test_acc - ablated_acc,
                }
            )
    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "mean_cka_to_final": float(np.mean(ckas)),
    }
    for index, value in enumerate(importance):
        metrics[f"component_{index}_importance"] = value
    return metrics, intervention_rows


def first_stable_step(
    rows: list[dict[str, float]], field: str, threshold: float
) -> int | None:
    """First checkpoint after which the criterion never fails again."""
    for index in range(len(rows)):
        if all(float(row[field]) >= threshold for row in rows[index:]):
            return int(rows[index]["step"])
    return None


def run_seed(config: Config, seed: int, output: Path) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x, train_y, test_x, test_y = make_data(config, seed)
    model = TinyTransformer(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoints: list[tuple[int, dict[str, torch.Tensor]]] = []

    for step in range(config.steps + 1):
        if step % config.checkpoint_every == 0:
            checkpoints.append(
                (step, {k: v.detach().clone() for k, v in model.state_dict().items()})
            )
        if step == config.steps:
            break
        model.train()
        logits, _ = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.load_state_dict(checkpoints[-1][1])
    model.eval()
    with torch.no_grad():
        _, final_acts = model(test_x)
    final_representations = [
        final_acts[f"layer_{layer}"][:, -1].clone()
        for layer in range(config.n_layers)
    ]

    rows: list[dict[str, float]] = []
    all_interventions: list[dict[str, float]] = []
    for step, state in checkpoints:
        metrics, interventions = evaluate_checkpoint(
            model, state, final_representations, train_x, train_y, test_x, test_y
        )
        metrics.update({"seed": seed, "step": step})
        rows.append(metrics)
        for intervention in interventions:
            intervention.update({"seed": seed, "step": step})
            all_interventions.append(intervention)

    write_csv(output / f"metrics_seed_{seed}.csv", rows)
    write_csv(output / f"interventions_seed_{seed}.csv", all_interventions)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return [
        {key: int(value) if key in {"seed", "step"} else float(value) for key, value in row.items()}
        for row in rows
    ]


def summarize_and_plot(
    config: Config, runs: dict[int, list[dict[str, float]]], output: Path
) -> dict:
    steps = np.array([row["step"] for row in next(iter(runs.values()))])
    acc = np.array([[row["test_accuracy"] for row in rows] for rows in runs.values()])
    cka = np.array([[row["mean_cka_to_final"] for row in rows] for rows in runs.values()])
    n_components = config.n_layers * (config.n_heads + config.mlp_groups)
    importance_keys = [f"component_{i}_importance" for i in range(n_components)]
    mechanism_similarity = []
    for rows in runs.values():
        vectors = np.array([[row[key] for key in importance_keys] for row in rows])
        final = vectors[-1]
        similarities = []
        for vector in vectors:
            if np.std(vector) < 1e-12 or np.std(final) < 1e-12:
                similarities.append(0.0)
            else:
                similarities.append(float(np.corrcoef(vector, final)[0, 1]))
        mechanism_similarity.append(similarities)
    mechanism_similarity = np.array(mechanism_similarity)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    panels = [
        (acc, "Accuracy on complete truth table", (0, 1.03)),
        (cka, "Representation similarity to final (CKA)", (0, 1.03)),
        (mechanism_similarity, "Causal intervention profile similarity to final", (-1.03, 1.03)),
    ]
    thresholds = (0.99, 0.95, 0.90)
    for axis, (values, label, ylim), threshold in zip(axes, panels, thresholds):
        mean = values.mean(0)
        spread = values.std(0)
        axis.plot(steps, mean, color="#d97757", linewidth=2)
        axis.fill_between(steps, mean - spread, mean + spread, color="#d97757", alpha=0.2)
        axis.axhline(threshold, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
        axis.set_xlabel("Training step")
        axis.set_ylabel(label)
        axis.set_ylim(*ylim)
        axis.set_xscale("symlog", linthresh=100)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "stability_timeline.png", dpi=220)
    fig.savefig(output / "stability_timeline.pdf")
    plt.close(fig)

    per_seed = []
    for seed, rows in runs.items():
        mech_rows = [
            {**row, "mechanism_similarity": float(mechanism_similarity[i, j])}
            for j, row in enumerate(rows)
            for i, candidate_seed in enumerate(runs)
            if candidate_seed == seed
        ]
        per_seed.append(
            {
                "seed": seed,
                "behavior_step": first_stable_step(rows, "test_accuracy", 0.99),
                "representation_step": first_stable_step(
                    rows, "mean_cka_to_final", 0.95
                ),
                "mechanism_step": first_stable_step(
                    mech_rows, "mechanism_similarity", 0.9
                ),
                "final_test_accuracy": rows[-1]["test_accuracy"],
            }
        )
    summary = {"config": asdict(config), "per_seed": per_seed}
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="One short smoke-test run")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--plot-only", action="store_true", help="Regenerate figures from saved metrics"
    )
    args = parser.parse_args()
    config = Config()
    if args.quick:
        config.steps = 100
        config.checkpoint_every = 50
        config.seeds = (0,)
    if args.steps is not None:
        config.steps = args.steps
    if args.seeds is not None:
        config.seeds = tuple(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with (args.output / "config.json").open() as handle:
            saved = json.load(handle)
        saved["seeds"] = tuple(saved["seeds"])
        config = Config(**saved)
        runs = {
            seed: read_metrics(args.output / f"metrics_seed_{seed}.csv")
            for seed in config.seeds
        }
        print(json.dumps(summarize_and_plot(config, runs, args.output), indent=2))
        return
    with (args.output / "config.json").open("w") as handle:
        json.dump(asdict(config), handle, indent=2)
    runs = {seed: run_seed(config, seed, args.output) for seed in config.seeds}
    summary = summarize_and_plot(config, runs, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
