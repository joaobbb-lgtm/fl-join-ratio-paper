#!/usr/bin/env python3
"""Analyze round-wise convergence of the 75 FashionMNIST FL executions.

Round 0 is the initial-model evaluation. Rounds 1..50 represent performance
after 1..50 completed global updates. This script analyzes convergence in
rounds; it deliberately does not label cumulative iteration time as exact
time-to-accuracy because the PFLlib loop evaluates before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = ("FedAvg", "FedProx", "SCAFFOLD")
JOIN_RATIOS = (0.10, 0.25, 0.50, 0.75, 1.00)
CHECKPOINT_ROUNDS = (10, 20, 30, 40, 50)
RELATIVE_LEVELS = (0.90, 0.95, 0.99)
ABSOLUTE_CANDIDATES = (0.60, 0.65, 0.70, 0.72, 0.74)
COLORS = {
    0.10: "#185FA5",
    0.25: "#378ADD",
    0.50: "#0F6E56",
    0.75: "#BA7517",
    1.00: "#993C1D",
}
ALGORITHM_COLORS = {"FedAvg": "#185FA5", "FedProx": "#0F6E56", "SCAFFOLD": "#993C1D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_inputs(rounds: pd.DataFrame, runs: pd.DataFrame) -> None:
    round_required = {"algorithm", "join_ratio", "rep", "round", "test_accuracy", "train_loss"}
    run_required = {"algorithm", "join_ratio", "rep", "best_accuracy_reported"}
    missing_round = sorted(round_required - set(rounds.columns))
    missing_run = sorted(run_required - set(runs.columns))
    if missing_round or missing_run:
        raise ValueError(f"Missing columns: rounds={missing_round}, runs={missing_run}")
    if len(runs) != 75 or len(rounds) != 75 * 51:
        raise ValueError(f"Expected 75 runs and 3825 round rows; found {len(runs)} and {len(rounds)}")
    counts = rounds.groupby(["algorithm", "join_ratio", "rep"])["round"].agg(["count", "min", "max"])
    if not ((counts["count"] == 51) & (counts["min"] == 0) & (counts["max"] == 50)).all():
        raise ValueError("Every execution must contain exactly rounds 0..50")
    config_counts = runs.groupby(["algorithm", "join_ratio"]).size()
    if len(config_counts) != 15 or not (config_counts == 5).all():
        raise ValueError("Expected five repetitions for each of 15 configurations")


def first_round_at_or_above(group: pd.DataFrame, threshold: float) -> float:
    reached = group.loc[group["test_accuracy"] >= threshold, "round"]
    return float(reached.min()) if not reached.empty else np.nan


def sustained_round_at_or_above(group: pd.DataFrame, threshold: float) -> float:
    """Earliest round after which every remaining evaluation stays above threshold."""
    ordered = group.sort_values("round")
    values = ordered["test_accuracy"].to_numpy(dtype=float)
    rounds = ordered["round"].to_numpy(dtype=int)
    suffix_min = np.minimum.accumulate(values[::-1])[::-1]
    positions = np.flatnonzero(suffix_min >= threshold)
    return float(rounds[positions[0]]) if positions.size else np.nan


def area_under_learning_curve(group: pd.DataFrame) -> float:
    ordered = group.sort_values("round")
    return float(np.trapezoid(ordered["test_accuracy"], ordered["round"]) / 50.0)


def choose_common_absolute_threshold(rounds: pd.DataFrame) -> float:
    best_by_run = rounds.groupby(["algorithm", "join_ratio", "rep"])["test_accuracy"].max()
    feasible = [value for value in ABSOLUTE_CANDIDATES if (best_by_run >= value).all()]
    if not feasible:
        raise ValueError("No configured absolute threshold is reached by all 75 runs")
    return max(feasible)


def build_run_metrics(rounds: pd.DataFrame, absolute_threshold: float) -> pd.DataFrame:
    rows = []
    for (algorithm, join_ratio, rep), group in rounds.groupby(
        ["algorithm", "join_ratio", "rep"], sort=True
    ):
        ordered = group.sort_values("round")
        accuracy = ordered.set_index("round")["test_accuracy"]
        best_accuracy = float(accuracy.max())
        best_round = int(accuracy.idxmax())
        row = {
            "algorithm": algorithm,
            "join_ratio": join_ratio,
            "rep": rep,
            "initial_accuracy_r0": float(accuracy.loc[0]),
            "final_accuracy_r50": float(accuracy.loc[50]),
            "best_accuracy_round_series": best_accuracy,
            "best_accuracy_round": best_round,
            "normalized_aulc": area_under_learning_curve(ordered),
            "gain_r0_to_r10": float(accuracy.loc[10] - accuracy.loc[0]),
            "gain_r0_to_r20": float(accuracy.loc[20] - accuracy.loc[0]),
            "mean_gain_per_round_r0_r10": float((accuracy.loc[10] - accuracy.loc[0]) / 10.0),
            "absolute_threshold": absolute_threshold,
            "first_round_absolute": first_round_at_or_above(ordered, absolute_threshold),
            "sustained_round_absolute": sustained_round_at_or_above(ordered, absolute_threshold),
        }
        for checkpoint in CHECKPOINT_ROUNDS:
            row[f"accuracy_round_{checkpoint}"] = float(accuracy.loc[checkpoint])
        for level in RELATIVE_LEVELS:
            label = int(level * 100)
            threshold = level * best_accuracy
            row[f"relative_threshold_{label}"] = threshold
            row[f"first_round_relative_{label}"] = first_round_at_or_above(ordered, threshold)
            row[f"sustained_round_relative_{label}"] = sustained_round_at_or_above(ordered, threshold)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["algorithm", "join_ratio", "rep"]).reset_index(drop=True)


def build_curve_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    return (
        rounds.groupby(["algorithm", "join_ratio", "round"], as_index=False)
        .agg(
            n=("rep", "size"),
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_median=("test_accuracy", "median"),
            accuracy_std=("test_accuracy", "std"),
            accuracy_q1=("test_accuracy", lambda x: x.quantile(0.25)),
            accuracy_q3=("test_accuracy", lambda x: x.quantile(0.75)),
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
        )
        .sort_values(["algorithm", "join_ratio", "round"])
        .reset_index(drop=True)
    )


def build_configuration_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "best_accuracy_round_series", "best_accuracy_round", "final_accuracy_r50",
        "normalized_aulc", "gain_r0_to_r10", "gain_r0_to_r20",
        "mean_gain_per_round_r0_r10", "first_round_absolute", "sustained_round_absolute",
        *[f"accuracy_round_{r}" for r in CHECKPOINT_ROUNDS],
        *[f"first_round_relative_{int(level * 100)}" for level in RELATIVE_LEVELS],
        *[f"sustained_round_relative_{int(level * 100)}" for level in RELATIVE_LEVELS],
    ]
    rows = []
    for (algorithm, join_ratio), group in run_metrics.groupby(["algorithm", "join_ratio"], sort=True):
        row = {"algorithm": algorithm, "join_ratio": join_ratio, "n": len(group)}
        for column in metric_columns:
            observed = group[column].dropna()
            row[f"{column}_mean"] = observed.mean() if not observed.empty else np.nan
            row[f"{column}_median"] = observed.median() if not observed.empty else np.nan
            row[f"{column}_std"] = observed.std(ddof=1) if len(observed) > 1 else np.nan
            row[f"{column}_reached_n"] = int(len(observed))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curves(curves: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=True)
    for axis, algorithm in zip(axes, ALGORITHMS):
        for join_ratio in JOIN_RATIOS:
            group = curves[
                (curves["algorithm"] == algorithm)
                & np.isclose(curves["join_ratio"], join_ratio)
            ].sort_values("round")
            x = group["round"].to_numpy(dtype=float)
            mean = group["accuracy_mean"].to_numpy(dtype=float)
            std = group["accuracy_std"].to_numpy(dtype=float)
            axis.plot(x, mean, color=COLORS[join_ratio], linewidth=1.7, label=f"q={join_ratio:.2f}")
            axis.fill_between(x, mean - std, mean + std, color=COLORS[join_ratio], alpha=0.10)
        axis.set_title(algorithm)
        axis.set_xlabel("Global updates completed")
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    axes[0].set_ylabel("Mean test accuracy")
    axes[-1].legend(fontsize=8, title="Join ratio", loc="lower right")
    fig.tight_layout()
    save_figure(fig, output_dir / "convergence_curves")


def plot_configuration_bars(summary: pd.DataFrame, column: str, ylabel: str,
                            stem: str, output_dir: Path) -> None:
    positions = np.arange(len(JOIN_RATIOS))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    for offset, algorithm in zip((-width, 0.0, width), ALGORITHMS):
        group = summary[summary["algorithm"] == algorithm].sort_values("join_ratio")
        axis.bar(
            positions + offset, group[column], width=width,
            label=algorithm, color=ALGORITHM_COLORS[algorithm], alpha=0.9,
        )
    axis.set_xticks(positions, [f"{q:.2f}" for q in JOIN_RATIOS])
    axis.set_xlabel("Join ratio")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    axis.legend(title="Algorithm")
    fig.tight_layout()
    save_figure(fig, output_dir / stem)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(run_metrics: pd.DataFrame, summary: pd.DataFrame,
                 absolute_threshold: float, output_dir: Path) -> None:
    lines = [
        "ROUND-WISE CONVERGENCE ANALYSIS",
        "Round 0: initial untrained-model evaluation",
        "Rounds 1..50: performance after 1..50 completed global updates",
        "Primary executions: 75",
        f"Common absolute threshold reached by all runs: {absolute_threshold:.2f}",
        "AULC: trapezoidal area under accuracy vs. global-update curve, divided by 50",
        "Relative thresholds: fractions of each run's own maximum logged accuracy",
        "",
        "CONFIGURATION SUMMARY",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.algorithm} | q={row.join_ratio:.2f} | "
            f"AULC_mean={row.normalized_aulc_mean:.6f} | "
            f"best_round_median={row.best_accuracy_round_median:.1f} | "
            f"first_r95_median={row.first_round_relative_95_median:.1f} | "
            f"first_abs_median={row.first_round_absolute_median:.1f} | "
            f"final_acc_mean={row.final_accuracy_r50_mean:.6f}"
        )
    lines.extend([
        "",
        "CAUTION",
        "First-hit rounds can precede later accuracy oscillations; sustained-round metrics are provided separately.",
        "No exact time-to-threshold or energy-to-threshold is claimed by this script because evaluation occurs before training inside each PFLlib loop iteration.",
    ])
    (output_dir / "convergence_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rounds = pd.read_csv(args.rounds)
    runs = pd.read_csv(args.runs)
    validate_inputs(rounds, runs)
    absolute_threshold = choose_common_absolute_threshold(rounds)
    run_metrics = build_run_metrics(rounds, absolute_threshold)
    curves = build_curve_summary(rounds)
    summary = build_configuration_summary(run_metrics)

    run_metrics.to_csv(args.output_dir / "convergence_per_run.csv", index=False)
    curves.to_csv(args.output_dir / "convergence_curves_summary.csv", index=False)
    summary.to_csv(args.output_dir / "convergence_configuration_summary.csv", index=False)
    plot_curves(curves, args.output_dir)
    plot_configuration_bars(
        summary, "normalized_aulc_mean", "Mean normalized area under learning curve",
        "aulc_by_configuration", args.output_dir,
    )
    plot_configuration_bars(
        summary, "first_round_relative_95_median", "Median first round reaching 95% of run maximum",
        "rounds_to_relative_95", args.output_dir,
    )
    write_report(run_metrics, summary, absolute_threshold, args.output_dir)

    print(f"input_runs: {len(runs)}")
    print(f"input_round_rows: {len(rounds)}")
    print(f"curve_summary_rows: {len(curves)}")
    print(f"configuration_rows: {len(summary)}")
    print(f"common_absolute_threshold: {absolute_threshold:.2f}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
