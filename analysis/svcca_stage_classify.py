"""Sweep SVCCA ranks and thresholds to report ReProF layer partitions.

The script reads COT-Bench through ``data/`` and extracts hidden states from
the pretrained feature extractor.  It never creates a classifier, fits a
ridge head, or evaluates accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.linalg import svd
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_manager import DataManager  # noqa: E402
from utils.inc_net import get_convnet  # noqa: E402


RANKS = (32, 64, 100, 128, 200, 256, 512)
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.60, 0.951, 0.05))


def reduce_dim(features: torch.Tensor, rank: int) -> torch.Tensor:
    """Return the rank-truncated left singular representation used by ReProF."""
    u, singular_values, _ = svd(features - features.mean(0, keepdim=True), full_matrices=False)
    return u[:, :rank] * singular_values[:rank]


def svcca_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    joint = torch.cat([left, right], dim=1).T
    covariance = torch.cov(joint)
    left_dim = left.shape[1]
    c_xx = covariance[:left_dim, :left_dim]
    c_yy = covariance[left_dim:, left_dim:]
    c_xy = covariance[:left_dim, left_dim:]
    epsilon = 1e-5
    c_xx = c_xx + epsilon * torch.eye(left_dim, device=left.device, dtype=left.dtype)
    c_yy = c_yy + epsilon * torch.eye(right.shape[1], device=right.device, dtype=right.dtype)
    inv_cxx = torch.cholesky_inverse(torch.linalg.cholesky(c_xx))
    inv_cyy = torch.cholesky_inverse(torch.linalg.cholesky(c_yy))
    canonical = inv_cxx @ c_xy @ inv_cyy @ c_xy.T
    values = torch.sqrt(torch.clamp(torch.linalg.eigvals(canonical).real, min=0))
    return values.mean().item()


def partition_layers(features: torch.Tensor, rank: int, threshold: float) -> tuple[list[list[int]], list[tuple[int, float, bool]]]:
    """Apply the greedy deep-to-shallow SVCCA grouping used by ReProF."""
    layer_count = features.shape[0]
    reduced = [reduce_dim(features[index], rank) for index in range(layer_count)]
    groups: list[list[int]] = []
    decisions: list[tuple[int, float, bool]] = []
    current_features = features[-1]
    current_reduced = reduced[-1]
    current_group = [layer_count - 1]

    for layer in range(layer_count - 2, -1, -1):
        similarity = svcca_correlation(current_reduced, reduced[layer])
        merge = similarity > threshold
        decisions.append((layer + 1, similarity, merge))  # one-based layer number
        if merge:
            current_features = (current_features * len(current_group) + features[layer]) / (len(current_group) + 1)
            current_reduced = reduce_dim(current_features, rank)
            current_group.append(layer)
        else:
            groups.append(sorted(layer + 1 for layer in current_group))
            current_features = features[layer]
            current_reduced = reduced[layer]
            current_group = [layer]
    groups.append(sorted(layer + 1 for layer in current_group))
    return groups, decisions


def format_groups(groups: list[list[int]]) -> str:
    # Report groups from shallow to deep, which is easier to read against layers 1--16.
    return " | ".join("{" + ", ".join(f"L{layer}" for layer in group) + "}" for group in reversed(groups))


def extract_features(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    data_manager = DataManager(
        args.dataset, args.model, shuffle=False, seed=args.seed,
        init_cls=args.init_cls, increment=args.increment,
    )
    dataset = data_manager.get_texts_dataset(np.arange(args.init_cls), source="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = get_convnet({"model_name": "ReProf", "convnet_type": args.model}).to(device).eval()
    per_layer: list[list[torch.Tensor]] | None = None

    with torch.inference_mode():
        for batch in loader:
            inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "output_hidden_states": True,
            }
            hidden_states = model(**inputs)
            if per_layer is None:
                per_layer = [[] for _ in hidden_states]
            for storage, hidden_state in zip(per_layer, hidden_states):
                storage.append(hidden_state.float().cpu())

    del model
    torch.cuda.empty_cache()
    assert per_layer is not None
    # The input embedding is not a transformer block and is excluded in ReProF.
    return torch.stack([torch.cat(items, dim=0) for items in per_layer], dim=0)[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Report SVCCA layer partitions without classifier training.")
    parser.add_argument("--dataset", default="data/COT-Bench/abstracts")
    parser.add_argument("--model", default="Llama-3.2-1B")
    parser.add_argument("--device", default="7")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-cls", type=int, default=2)
    parser.add_argument("--increment", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/figure07_svcca_parameter_sensitivity/groups.log"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to extract the Llama feature tensors.")
    device = torch.device(f"cuda:{args.device}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "SVCCA layer-partition sensitivity (no classifier training)",
        f"dataset: {args.dataset}",
        f"feature extractor: {args.model}",
        f"source: initial training task (classes 0-1), seed={args.seed}",
        f"SVD ranks: {list(RANKS)}",
        f"thresholds: {list(THRESHOLDS)}",
        "",
        "Extracting hidden states from data/...",
    ]
    print("\n".join(lines), flush=True)
    features = extract_features(args, device)
    features = features.to(device)
    header = f"Extracted {features.shape[0]} transformer layers for {features.shape[1]} training samples."
    print(header, flush=True)
    lines.append(header)
    lines.append("")

    for rank in RANKS:
        print(f"\nSVD rank r={rank}", flush=True)
        lines.append(f"SVD rank r={rank}")
        for threshold in THRESHOLDS:
            groups, decisions = partition_layers(features, rank, threshold)
            group_text = format_groups(groups)
            decision_text = "; ".join(
                f"L{layer}:{similarity:.4f}{' merge' if merge else ' split'}"
                for layer, similarity, merge in decisions
            )
            result = f"  tau={threshold:.2f} | {len(groups)} blocks | {group_text}"
            print(result, flush=True)
            lines.extend([result, f"    decisions: {decision_text}"])
        lines.append("")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
