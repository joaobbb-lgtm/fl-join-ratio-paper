#!/usr/bin/env python3
"""Quantify marginal returns of increasing the FL join ratio.

The analysis uses the 75 validated FashionMNIST runs. Accuracy is aggregated
by the mean, while time and estimated energy use the median, matching the
reporting convention of the LANC 2026 paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = ("FedAvg", "FedProx", "SCAFFOLD")
JOIN_RATIOS = (0.10, 0.25, 0.50, 0.75, 1.00)
COLORS = {"FedAvg": "#185FA5", "FedProx": "#0F6E56", "SCAFFOLD": "#993C1D"}
MARKERS = {"FedAvg": "o", "FedProx": "s", "SCAFFOLD": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "algorithm", "join_ratio", "rep", "best_accuracy_reported",
        "total_time_reported_s", "estimated_energy_j",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    if len(frame) != 75:
        raise ValueError(f"Expected 75 primary runs, found {len(frame)}")
    counts = frame.groupby(["algorithm", "join_ratio"]).size()
    if len(counts) != 15 or not (counts == 5).all():
        raise ValueError("Expected five runs in each of the 15 configurations")


def aggregate_configurations(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.groupby(["algorithm", "join_ratio"], as_index=False)
        .agg(
            n=("rep", "size"),
            accuracy_mean=("best_accuracy_reported", "mean"),
            accuracy_median=("best_accuracy_reported", "median"),
            accuracy_std=("best_accuracy_reported", "std"),
            time_median_s=("total_time_reported_s", "median"),
            energy_median_j=("estimated_energy_j", "median"),
        )
        .sort_values(["algorithm", "join_ratio"])
        .reset_index(drop=True)
    )
    result["energy_median_kj"] = result["energy_median_j"] / 1000.0
    result["energy_efficiency_acc_per_kj"] = (
        result["accuracy_mean"] / result["energy_median_kj"]
    )
    return result


def safe_positive_ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0:
        return np.nan
    return numerator / denominator


def marginal_table(configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHMS:
        group = configs[configs["algorithm"] == algorithm].sort_values("join_ratio")
        records = group.to_dict("records")
        for previous, current in zip(records[:-1], records[1:]):
            delta_acc = current["accuracy_mean"] - previous["accuracy_mean"]
            delta_acc_pp = 100.0 * delta_acc
            delta_time = current["time_median_s"] - previous["time_median_s"]
            delta_energy = current["energy_median_kj"] - previous["energy_median_kj"]
            rows.append({
                "algorithm": algorithm,
                "join_ratio_from": previous["join_ratio"],
                "join_ratio_to": current["join_ratio"],
                "transition": f"{previous['join_ratio']:.2f}->{current['join_ratio']:.2f}",
                "accuracy_from": previous["accuracy_mean"],
                "accuracy_to": current["accuracy_mean"],
                "delta_accuracy": delta_acc,
                "delta_accuracy_percentage_points": delta_acc_pp,
                "time_from_s": previous["time_median_s"],
                "time_to_s": current["time_median_s"],
                "delta_time_s": delta_time,
                "time_increase_percent": 100.0 * delta_time / previous["time_median_s"],
                "energy_from_kj": previous["energy_median_kj"],
                "energy_to_kj": current["energy_median_kj"],
                "delta_energy_kj": delta_energy,
                "energy_increase_percent": 100.0 * delta_energy / previous["energy_median_kj"],
                "accuracy_pp_per_1000_additional_s": (
                    1000.0 * delta_acc_pp / delta_time if delta_time > 0 else np.nan
                ),
                "accuracy_pp_per_additional_kj": (
                    delta_acc_pp / delta_energy if delta_energy > 0 else np.nan
                ),
                "additional_s_per_0_01_accuracy": safe_positive_ratio(0.01 * delta_time, delta_acc),
                "additional_kj_per_0_01_accuracy": safe_positive_ratio(0.01 * delta_energy, delta_acc),
                "nonpositive_accuracy_gain": bool(delta_acc <= 0),
            })
    return pd.DataFrame(rows)


def cumulative_table(configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in ALGORITHMS:
        group = configs[configs["algorithm"] == algorithm].sort_values("join_ratio")
        baseline = group.iloc[0]
        for row in group.itertuples(index=False):
            rows.append({
                "algorithm": algorithm,
                "join_ratio": row.join_ratio,
                "baseline_join_ratio": baseline["join_ratio"],
                "accuracy_gain_from_baseline": row.accuracy_mean - baseline["accuracy_mean"],
                "accuracy_gain_pp_from_baseline": 100.0 * (row.accuracy_mean - baseline["accuracy_mean"]),
                "additional_time_s_from_baseline": row.time_median_s - baseline["time_median_s"],
                "additional_energy_kj_from_baseline": row.energy_median_kj - baseline["energy_median_kj"],
                "time_ratio_to_baseline": row.time_median_s / baseline["time_median_s"],
                "energy_ratio_to_baseline": row.energy_median_kj / baseline["energy_median_kj"],
            })
    return pd.DataFrame(rows)


def knee_candidates(configs: pd.DataFrame) -> pd.DataFrame:
    """Exploratory maximum-distance-to-chord knee for each cost measure."""
    rows = []
    for algorithm in ALGORITHMS:
        group = configs[configs["algorithm"] == algorithm].sort_values("join_ratio").reset_index(drop=True)
        for cost_name, cost_column in (("time", "time_median_s"), ("energy", "energy_median_kj")):
            x = group[cost_column].to_numpy(dtype=float)
            y = group["accuracy_mean"].to_numpy(dtype=float)
            x_norm = (x - x.min()) / (x.max() - x.min())
            y_norm = (y - y.min()) / (y.max() - y.min())
            start = np.array([x_norm[0], y_norm[0]])
            end = np.array([x_norm[-1], y_norm[-1]])
            chord = end - start
            offsets = np.column_stack([x_norm, y_norm]) - start
            cross_2d = chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
            distances = np.abs(cross_2d) / np.linalg.norm(chord)
            distances[[0, -1]] = 0.0
            index = int(np.argmax(distances))
            rows.append({
                "algorithm": algorithm,
                "cost_dimension": cost_name,
                "candidate_join_ratio": group.loc[index, "join_ratio"],
                "normalized_distance_to_chord": distances[index],
                "accuracy_mean": group.loc[index, "accuracy_mean"],
                "cost_value": group.loc[index, cost_column],
                "exploratory_only": True,
            })
    return pd.DataFrame(rows)


def plot_accuracy(configs: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for algorithm in ALGORITHMS:
        group = configs[configs["algorithm"] == algorithm].sort_values("join_ratio")
        axis.errorbar(
            group["join_ratio"], group["accuracy_mean"], yerr=group["accuracy_std"],
            label=algorithm, color=COLORS[algorithm], marker=MARKERS[algorithm],
            linewidth=2, markersize=6, capsize=3,
        )
    axis.set_xlabel("Join ratio")
    axis.set_ylabel("Mean best test accuracy")
    axis.set_xticks(JOIN_RATIOS)
    axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    axis.legend(title="Algorithm")
    fig.tight_layout()
    save_figure(fig, output_dir / "accuracy_by_join_ratio")


def plot_marginal(marginal: pd.DataFrame, value_column: str, ylabel: str, stem: str, output_dir: Path) -> None:
    transitions = marginal["transition"].drop_duplicates().tolist()
    positions = np.arange(len(transitions))
    width = 0.24
    fig, axis = plt.subplots(figsize=(7.8, 4.4))
    for offset, algorithm in zip((-width, 0.0, width), ALGORITHMS):
        group = marginal[marginal["algorithm"] == algorithm].set_index("transition").loc[transitions]
        axis.bar(positions + offset, group[value_column], width=width,
                 label=algorithm, color=COLORS[algorithm], alpha=0.9)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, transitions)
    axis.set_xlabel("Join-ratio transition")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    axis.legend(title="Algorithm")
    fig.tight_layout()
    save_figure(fig, output_dir / stem)


def save_figure(fig: plt.Figure, path_without_extension: Path) -> None:
    fig.savefig(path_without_extension.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_without_extension.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(configs: pd.DataFrame, marginal: pd.DataFrame, knees: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "MARGINAL RETURNS ANALYSIS",
        "Aggregation: mean best accuracy; median execution time; median estimated energy",
        "Transitions: consecutive observed join ratios",
        "Knee method: exploratory maximum normalized distance to the endpoint chord",
        "",
        "MARGINAL RESULTS",
    ]
    for row in marginal.itertuples(index=False):
        lines.append(
            f"{row.algorithm} | {row.transition} | delta_acc_pp={row.delta_accuracy_percentage_points:+.6f} | "
            f"delta_time_s={row.delta_time_s:+.3f} | delta_energy_kj={row.delta_energy_kj:+.6f} | "
            f"pp_per_1000s={row.accuracy_pp_per_1000_additional_s:+.6f} | "
            f"pp_per_kj={row.accuracy_pp_per_additional_kj:+.6f}"
        )
    lines.extend(["", "EXPLORATORY KNEE CANDIDATES"])
    for row in knees.itertuples(index=False):
        lines.append(
            f"{row.algorithm} | {row.cost_dimension} | q={row.candidate_join_ratio:.2f} | "
            f"distance={row.normalized_distance_to_chord:.6f}"
        )
    lines.extend([
        "",
        "CAUTION",
        "Knee candidates are descriptive points within a five-level experimental grid, not estimated universal optima.",
    ])
    (output_dir / "marginal_returns_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(args.runs)
    validate_input(runs)
    configs = aggregate_configurations(runs)
    marginal = marginal_table(configs)
    cumulative = cumulative_table(configs)
    knees = knee_candidates(configs)

    configs.to_csv(args.output_dir / "configuration_aggregates.csv", index=False)
    marginal.to_csv(args.output_dir / "marginal_returns.csv", index=False)
    cumulative.to_csv(args.output_dir / "cumulative_from_q010.csv", index=False)
    knees.to_csv(args.output_dir / "knee_candidates.csv", index=False)

    plot_accuracy(configs, args.output_dir)
    plot_marginal(marginal, "delta_accuracy_percentage_points", "Accuracy gain (percentage points)",
                  "marginal_accuracy_gain", args.output_dir)
    plot_marginal(marginal, "accuracy_pp_per_1000_additional_s", "Accuracy gain (pp) per 1,000 additional s",
                  "marginal_gain_per_time", args.output_dir)
    plot_marginal(marginal, "accuracy_pp_per_additional_kj", "Accuracy gain (pp) per additional kJ",
                  "marginal_gain_per_energy", args.output_dir)
    write_report(configs, marginal, knees, args.output_dir)

    print(f"input_runs: {len(runs)}")
    print(f"configuration_rows: {len(configs)}")
    print(f"marginal_transitions: {len(marginal)}")
    print(f"nonpositive_accuracy_gains: {int(marginal['nonpositive_accuracy_gain'].sum())}")
    print(f"knee_candidates: {len(knees)}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
