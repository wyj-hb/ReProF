"""Reproduce Table 9: stage-wise SVCCA layer-block partitions.

The script reads the cached per-layer feature tensors that were used for the
original analysis.  It reports only the ten stage partitions, the pooled
partition, and Rand Index against the stage-1 partition.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_ROOT = Path("/home/wangyujie/code/mutil_ranpac/Abstracts/train")
DEFAULT_OUTPUT = ROOT / "results/table09_stage_partition_stability/stage_partition_stability.log"
LAYERS = tuple(range(1, 17))
STAGES = tuple(range(1, 11))
RANK = 100
THRESHOLD = 0.80


def stage_name(stage: int) -> str:
    start = (stage - 1) * 2
    return f"{start}-{start + 2}-train_Features_h.pt"


def reduce_dim(features: torch.Tensor, rank: int) -> torch.Tensor:
    u, singular_values, _ = torch.linalg.svd(
        features - features.mean(0, keepdim=True), full_matrices=False
    )
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


def partition_layers(features: Sequence[torch.Tensor], rank: int, threshold: float) -> list[list[int]]:
    """Greedy deep-to-shallow partition; returned groups are zero-based."""
    reduced = [reduce_dim(layer, rank) for layer in features]
    groups: list[list[int]] = []
    current_features = features[-1]
    current_reduced = reduced[-1]
    current_group = [len(features) - 1]

    for layer in range(len(features) - 2, -1, -1):
        if svcca_correlation(current_reduced, reduced[layer]) > threshold:
            current_features = (current_features * len(current_group) + features[layer]) / (len(current_group) + 1)
            current_reduced = reduce_dim(current_features, rank)
            current_group.append(layer)
        else:
            groups.append(sorted(current_group))
            current_features = features[layer]
            current_reduced = reduced[layer]
            current_group = [layer]
    groups.append(sorted(current_group))
    return groups


def labels(groups: Sequence[Sequence[int]]) -> np.ndarray:
    result = np.full(len(LAYERS), -1, dtype=np.int32)
    for index, group in enumerate(groups):
        result[list(group)] = index
    return result


def rand_index(reference: np.ndarray, candidate: np.ndarray) -> float:
    rows, cols = np.triu_indices(len(reference), k=1)
    return float(np.mean((reference[rows] == reference[cols]) == (candidate[rows] == candidate[cols])))


def format_group(group: Sequence[int]) -> str:
    first, last = group[0] + 1, group[-1] + 1
    return f"L{first}" if first == last else f"L{first}–L{last}"


def format_partition(groups: Sequence[Sequence[int]]) -> str:
    # partition_layers is deepest-to-shallowest; Table 9 is shallow-to-deep.
    return " | ".join(format_group(group) for group in reversed(groups))


def load_stage(features_root: Path, stage: int, device: torch.device) -> list[torch.Tensor]:
    filename = stage_name(stage)
    tensors = []
    for layer in LAYERS:
        path = features_root / f"layer_{layer}" / "features" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached feature tensor: {path}")
        tensors.append(torch.load(path, map_location=device, weights_only=True))
    return tensors


def load_pooled(features_root: Path, device: torch.device) -> list[torch.Tensor]:
    pooled: list[torch.Tensor] = []
    for layer in LAYERS:
        tensors = [
            torch.load(features_root / f"layer_{layer}" / "features" / stage_name(stage), map_location="cpu", weights_only=True)
            for stage in STAGES
        ]
        pooled.append(torch.cat(tensors, dim=0).to(device))
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description="Report stage-wise SVCCA partitions for Table 9.")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--device", default="7")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SVCCA computation.")
    device = torch.device(f"cuda:{args.device}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write(message: str = "") -> None:
        print(message, flush=True)
        with args.output.open("a", encoding="utf-8") as output:
            output.write(message + "\n")

    args.output.write_text("", encoding="utf-8")
    write("Stage-wise SVCCA layer-block partition stability (Table 9)")
    write(f"cached features: {args.features_root}")
    write(f"feature extractor: Llama-3.2-1B; layers=16; r={RANK}; tau={THRESHOLD:.2f}")
    write(f"device: cuda:{args.device}")
    write("")
    write("Stage-wise partitions (Table 9 order: shallow to deep)")

    groups_by_stage: dict[int, list[list[int]]] = {}
    labels_by_stage: dict[int, np.ndarray] = {}
    for stage in STAGES:
        stage_features = load_stage(args.features_root, stage, device)
        groups = partition_layers(stage_features, RANK, THRESHOLD)
        groups_by_stage[stage] = groups
        labels_by_stage[stage] = labels(groups)
        write(f"Stage {stage}: {format_partition(groups)}")
        del stage_features
        torch.cuda.empty_cache()

    write("")
    write("Pooled partition")
    pooled_features = load_pooled(args.features_root, device)
    pooled_groups = partition_layers(pooled_features, RANK, THRESHOLD)
    write(f"All stages: {format_partition(pooled_groups)}")
    del pooled_features
    torch.cuda.empty_cache()

    reference = labels_by_stage[1]
    scores = {stage: rand_index(reference, stage_labels) for stage, stage_labels in labels_by_stage.items()}
    write("")
    write("Rand Index relative to Stage 1")
    for stage in STAGES:
        write(f"Stage {stage}: {scores[stage]:.4f}")
    write(f"Mean, Stages 2–10: {np.mean([scores[stage] for stage in STAGES[1:]]):.4f}")
    write(f"Mean, Stages 1–10: {np.mean(list(scores.values())):.4f}")
    write("")
    write(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
