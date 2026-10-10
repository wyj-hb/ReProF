"""Visualize layer-wise SVCCA similarity from cached feature tensors.

Each input file is a ``torch.save`` tensor of shape
``[num_layers, num_examples, hidden_size]``.  Features should be extracted
from the first incremental stage before running this utility; for Figure 3,
that is the first-stage samples in ``data/COT-Bench/abstracts/train.jsonl``.
The embedding layer is not included in the tensor.

Example:
    python utils/visualize_layer_similarity.py \
        --features features/Llama-3.2-1B.pt features/Qwen2.5-0.5B.pt \
        --titles Llama-3.2-1B Qwen2.5-0.5B-Instruct \
        --output results/figure03_layer_similarity.pdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def _reduce_features(features: torch.Tensor, components: int) -> torch.Tensor:
    """Project one layer's features to its leading singular components."""
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    max_components = min(components, centered.shape[0], centered.shape[1])
    if max_components < 1:
        raise ValueError("Each layer needs at least one example and one feature dimension.")
    u, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    return u[:, :max_components] * singular_values[:max_components]


def _svcca_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> float:
    """Return the mean canonical correlation of two reduced feature matrices."""
    joint = torch.cat([x, y], dim=1).T
    covariance = torch.cov(joint)
    x_dim = x.shape[1]
    c_xx = covariance[:x_dim, :x_dim]
    c_yy = covariance[x_dim:, x_dim:]
    c_xy = covariance[:x_dim, x_dim:]

    c_xx = c_xx + eps * torch.eye(x_dim, dtype=c_xx.dtype)
    y_dim = y.shape[1]
    c_yy = c_yy + eps * torch.eye(y_dim, dtype=c_yy.dtype)
    inv_xx = torch.linalg.inv(c_xx)
    inv_yy = torch.linalg.inv(c_yy)
    matrix = inv_xx @ c_xy @ inv_yy @ c_xy.T
    values = torch.linalg.eigvals(matrix).real.clamp_min(0).sqrt()
    return float(values.mean())


def compute_similarity_matrix(features: torch.Tensor, components: int = 100) -> np.ndarray:
    """Compute a layer-by-layer SVCCA similarity matrix for one extractor."""
    if features.ndim != 3:
        raise ValueError(
            f"Expected a [layers, examples, hidden_size] tensor, got {tuple(features.shape)}."
        )

    features = features.detach().cpu()
    reduced = [_reduce_features(layer, components) for layer in features]
    num_layers = len(reduced)
    similarities = torch.empty((num_layers, num_layers), dtype=torch.float32)
    for source in range(num_layers):
        for target in range(source, num_layers):
            value = _svcca_correlation(reduced[source], reduced[target])
            similarities[source, target] = value
            similarities[target, source] = value
    return similarities.numpy()


def plot_layer_similarity(
    feature_paths: Sequence[Path],
    output_path: Path,
    titles: Sequence[str] | None = None,
    components: int = 100,
    cmap: str = "coolwarm",
) -> list[np.ndarray]:
    """Load feature tensors, compute their SVCCA matrices, and save one PDF."""
    if titles is not None and len(titles) != len(feature_paths):
        raise ValueError("The number of --titles values must match the number of --features files.")

    matrices = []
    for feature_path in feature_paths:
        tensor = torch.load(feature_path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{feature_path} does not contain a tensor.")
        matrices.append(compute_similarity_matrix(tensor, components))

    columns = min(3, len(matrices))
    rows = math.ceil(len(matrices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5.5 * rows), squeeze=False)
    flat_axes = axes.ravel()
    lower = min(matrix.min() for matrix in matrices)
    upper = max(matrix.max() for matrix in matrices)

    for index, (axis, matrix) in enumerate(zip(flat_axes, matrices)):
        heatmap = sns.heatmap(matrix, ax=axis, cmap=cmap, vmin=lower, vmax=upper, cbar=False)
        layer_count = matrix.shape[0]
        tick_step = max(1, math.ceil(layer_count / 6))
        ticks = list(range(0, layer_count, tick_step))
        if ticks[-1] != layer_count - 1:
            ticks.append(layer_count - 1)
        positions = np.asarray(ticks) + 0.5
        axis.set_xticks(positions, ticks)
        axis.set_yticks(positions, ticks)
        axis.set_xlabel("Layer")
        axis.set_ylabel("Layer")
        axis.set_title(titles[index] if titles else feature_paths[index].stem)

    for axis in flat_axes[len(matrices):]:
        axis.remove()
    figure.colorbar(heatmap.collections[0], ax=flat_axes[:len(matrices)].tolist(), shrink=0.85)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    return matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate layer-wise SVCCA heatmaps.")
    parser.add_argument("--features", nargs="+", required=True, type=Path, help="Input .pt feature tensors.")
    parser.add_argument("--titles", nargs="+", help="Optional titles, one per feature tensor.")
    parser.add_argument("--output", required=True, type=Path, help="Output PDF path.")
    parser.add_argument("--components", type=int, default=100, help="Number of singular components for SVCCA.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    plot_layer_similarity(arguments.features, arguments.output, arguments.titles, arguments.components)
