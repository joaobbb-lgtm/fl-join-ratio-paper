#!/usr/bin/env python3
"""Analyze accuracy-cost Pareto frontiers for the 15 FL configurations.

Quality is maximized; execution time and estimated energy are minimized.
Balanced recommendations use equal-weight normalized Euclidean distance to the
ideal point and are reported separately from the Pareto classification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = ("FedAvg", "FedProx", "SCAFFOLD")
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
        raise ValueError(f"Missing columns: {missing}")
    if len(frame) != 75:
        raise ValueError(f"Expected 75 primary runs, found {len(frame)}")
    counts = frame.groupby(["algorithm", "join_ratio"]).size()
    if len(counts) != 15 or not (counts == 5).all():
        raise ValueError("Expected five runs in each of the 15 configurations")


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    configs = (
        frame.groupby(["algorithm", "join_ratio"], as_index=False)
        .agg(
            accuracy_mean=("best_accuracy_reported", "mean"),
            accuracy_std=("best_accuracy_reported", "std"),
            time_median_s=("total_time_reported_s", "median"),
            energy_median_j=("estimated_energy_j", "median"),
        )
        .sort_values(["algorithm", "join_ratio"])
        .reset_index(drop=True)
    )
    configs["energy_median_kj"] = configs["energy_median_j"] / 1000.0
    configs["energy_efficiency_acc_per_kj"] = configs["accuracy_mean"] / configs["energy_median_kj"]
    configs["time_efficiency_acc_per_1000s"] = 1000.0 * configs["accuracy_mean"] / configs["time_median_s"]
    configs["configuration"] = configs.apply(
        lambda row: f"{row['algorithm']}_q{row['join_ratio']:.2f}", axis=1
    )
    return configs


def dominates(a: pd.Series, b: pd.Series, cost_columns: tuple[str, ...]) -> bool:
    no_worse = a["accuracy_mean"] >= b["accuracy_mean"] and all(
        a[column] <= b[column] for column in cost_columns
    )
    strictly_better = a["accuracy_mean"] > b["accuracy_mean"] or any(
        a[column] < b[column] for column in cost_columns
    )
    return bool(no_worse and strictly_better)


def pareto_mask(frame: pd.DataFrame, cost_columns: tuple[str, ...]) -> pd.Series:
    flags = []
    for index, candidate in frame.iterrows():
        is_dominated = any(
            dominates(other, candidate, cost_columns)
            for other_index, other in frame.iterrows()
            if other_index != index
        )
        flags.append(not is_dominated)
    return pd.Series(flags, index=frame.index)


def add_pareto_flags(configs: pd.DataFrame) -> pd.DataFrame:
    result = configs.copy()
    result["pareto_accuracy_time_global"] = pareto_mask(result, ("time_median_s",))
    result["pareto_accuracy_energy_global"] = pareto_mask(result, ("energy_median_kj",))
    result["pareto_accuracy_time_energy_global"] = pareto_mask(
        result, ("time_median_s", "energy_median_kj")
    )
    for name, costs in (
        ("pareto_accuracy_time_within_algorithm", ("time_median_s",)),
        ("pareto_accuracy_energy_within_algorithm", ("energy_median_kj",)),
    ):
        result[name] = False
        for algorithm in ALGORITHMS:
            indices = result.index[result["algorithm"] == algorithm]
            result.loc[indices, name] = pareto_mask(result.loc[indices], costs).to_numpy()
    return result


def dominance_relations(configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dimensions = {
        "accuracy_time": ("time_median_s",),
        "accuracy_energy": ("energy_median_kj",),
        "accuracy_time_energy": ("time_median_s", "energy_median_kj"),
    }
    for dimension, costs in dimensions.items():
        for i, dominator in configs.iterrows():
            for j, dominated in configs.iterrows():
                if i != j and dominates(dominator, dominated, costs):
                    rows.append({
                        "dimension": dimension,
                        "dominator": dominator["configuration"],
                        "dominated": dominated["configuration"],
                    })
    return pd.DataFrame(rows)


def minmax(series: pd.Series) -> pd.Series:
    spread = series.max() - series.min()
    if spread == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / spread


def add_ideal_distances(configs: pd.DataFrame) -> pd.DataFrame:
    result = configs.copy()
    result["normalized_quality_loss"] = minmax(result["accuracy_mean"].max() - result["accuracy_mean"])
    result["normalized_time_cost"] = minmax(result["time_median_s"])
    result["normalized_energy_cost"] = minmax(result["energy_median_kj"])
    result["ideal_distance_accuracy_time"] = np.sqrt(
        (result["normalized_quality_loss"] ** 2 + result["normalized_time_cost"] ** 2) / 2.0
    )
    result["ideal_distance_accuracy_energy"] = np.sqrt(
        (result["normalized_quality_loss"] ** 2 + result["normalized_energy_cost"] ** 2) / 2.0
    )
    result["ideal_distance_accuracy_time_energy"] = np.sqrt(
        (
            result["normalized_quality_loss"] ** 2
            + result["normalized_time_cost"] ** 2
            + result["normalized_energy_cost"] ** 2
        ) / 3.0
    )
    return result


def recommendations(configs: pd.DataFrame) -> pd.DataFrame:
    rules = [
        ("maximum_quality", "accuracy_mean", "max", None),
        ("minimum_time", "time_median_s", "min", None),
        ("maximum_energy_efficiency", "energy_efficiency_acc_per_kj", "max", None),
        ("balanced_accuracy_time", "ideal_distance_accuracy_time", "min", "pareto_accuracy_time_global"),
        ("balanced_accuracy_energy", "ideal_distance_accuracy_energy", "min", "pareto_accuracy_energy_global"),
        (
            "balanced_accuracy_time_energy",
            "ideal_distance_accuracy_time_energy",
            "min",
            "pareto_accuracy_time_energy_global",
        ),
    ]
    rows = []
    for category, criterion, direction, pareto_column in rules:
        candidates = configs if pareto_column is None else configs[configs[pareto_column]]
        index = candidates[criterion].idxmax() if direction == "max" else candidates[criterion].idxmin()
        row = configs.loc[index]
        rows.append({
            "category": category,
            "configuration": row["configuration"],
            "algorithm": row["algorithm"],
            "join_ratio": row["join_ratio"],
            "accuracy_mean": row["accuracy_mean"],
            "time_median_s": row["time_median_s"],
            "energy_median_kj": row["energy_median_kj"],
            "energy_efficiency_acc_per_kj": row["energy_efficiency_acc_per_kj"],
            "criterion": criterion,
            "criterion_value": row[criterion],
            "equal_weights_when_balanced": category.startswith("balanced_"),
        })
    return pd.DataFrame(rows)


def plot_frontier(configs: pd.DataFrame, cost_column: str, cost_label: str,
                  pareto_column: str, stem: str, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 5.0))
    for algorithm in ALGORITHMS:
        group = configs[configs["algorithm"] == algorithm]
        axis.scatter(
            group[cost_column], group["accuracy_mean"], label=algorithm,
            color=COLORS[algorithm], marker=MARKERS[algorithm], s=58, zorder=3,
        )
        for row in group.itertuples(index=False):
            axis.annotate(
                f"q={row.join_ratio:.2f}",
                (getattr(row, cost_column), row.accuracy_mean),
                xytext=(4, 5), textcoords="offset points", fontsize=7,
            )
    frontier = configs[configs[pareto_column]].sort_values(cost_column)
    axis.plot(frontier[cost_column], frontier["accuracy_mean"], color="black",
              linestyle="--", linewidth=1.2, label="Global Pareto frontier", zorder=2)
    axis.set_xlabel(cost_label)
    axis.set_ylabel("Mean best test accuracy")
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(configs: pd.DataFrame, recs: pd.DataFrame, relations: pd.DataFrame,
                 output_dir: Path) -> None:
    lines = [
        "PARETO AND QUALITY-COST TRADE-OFF ANALYSIS",
        "Quality objective: maximize mean best accuracy",
        "Cost objectives: minimize median execution time and median estimated energy",
        "Balanced rule: minimum normalized Euclidean distance to the ideal point with equal weights",
        "",
    ]
    for label, column in (
        ("ACCURACY-TIME GLOBAL FRONTIER", "pareto_accuracy_time_global"),
        ("ACCURACY-ENERGY GLOBAL FRONTIER", "pareto_accuracy_energy_global"),
        ("THREE-OBJECTIVE GLOBAL FRONTIER", "pareto_accuracy_time_energy_global"),
    ):
        lines.append(label)
        for value in configs.loc[configs[column], "configuration"]:
            lines.append(value)
        lines.append("")
    lines.append("RECOMMENDATIONS")
    for row in recs.itertuples(index=False):
        lines.append(
            f"{row.category} | {row.configuration} | accuracy={row.accuracy_mean:.6f} | "
            f"time_s={row.time_median_s:.3f} | energy_kj={row.energy_median_kj:.6f}"
        )
    lines.extend([
        "",
        f"dominance_relations_total: {len(relations)}",
        "CAUTION",
        "Balanced recommendations depend on the explicitly stated equal-weight normalization and are not universal optima.",
    ])
    (output_dir / "pareto_tradeoff_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(args.runs)
    validate_input(runs)
    configs = add_ideal_distances(add_pareto_flags(aggregate(runs)))
    relations = dominance_relations(configs)
    recs = recommendations(configs)

    configs.to_csv(args.output_dir / "pareto_configurations.csv", index=False)
    relations.to_csv(args.output_dir / "dominance_relations.csv", index=False)
    recs.to_csv(args.output_dir / "recommendations.csv", index=False)
    plot_frontier(
        configs, "time_median_s", "Median execution time (s)",
        "pareto_accuracy_time_global", "pareto_accuracy_time", args.output_dir,
    )
    plot_frontier(
        configs, "energy_median_kj", "Median estimated energy (kJ)",
        "pareto_accuracy_energy_global", "pareto_accuracy_energy", args.output_dir,
    )
    write_report(configs, recs, relations, args.output_dir)

    print(f"input_runs: {len(runs)}")
    print(f"configurations: {len(configs)}")
    print(f"pareto_time_global: {int(configs['pareto_accuracy_time_global'].sum())}")
    print(f"pareto_energy_global: {int(configs['pareto_accuracy_energy_global'].sum())}")
    print(f"pareto_three_objective_global: {int(configs['pareto_accuracy_time_energy_global'].sum())}")
    print(f"recommendations: {len(recs)}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
