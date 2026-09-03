#!/usr/bin/env python3
"""Assess stability across the five repetitions of each LANC configuration.

No observation is automatically removed. Tukey fences are used only to flag
potential outliers for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = ["FedAvg", "FedProx", "SCAFFOLD"]
JOIN_RATIOS = [0.10, 0.25, 0.50, 0.75, 1.00]
METRICS = ["best_accuracy", "execution_time_s", "estimated_energy_kj"]
METRIC_LABELS = {
    "best_accuracy": "Best accuracy",
    "execution_time_s": "Execution time (s)",
    "estimated_energy_kj": "Estimated energy (kJ)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def first_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Expected one of {candidates}; found {list(df.columns)}")


def load_runs(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    algorithm = first_column(raw, ["algorithm", "algo"])
    join_ratio = first_column(raw, ["join_ratio", "ratio"])
    repetition = first_column(raw, ["rep", "repetition", "seed"])
    accuracy = first_column(
        raw,
        ["best_accuracy_from_rounds", "best_accuracy_reported", "best_accuracy"],
    )
    time = first_column(
        raw,
        ["total_time_reported_s", "execution_time_s", "total_time_s", "time_s"],
    )
    energy = first_column(
        raw,
        ["estimated_energy_kj", "energy_kj", "estimated_energy_j", "energy_j"],
    )
    runs = pd.DataFrame(
        {
            "algorithm": raw[algorithm].astype(str).str.strip(),
            "join_ratio": pd.to_numeric(raw[join_ratio], errors="raise"),
            "rep": raw[repetition],
            "best_accuracy": pd.to_numeric(raw[accuracy], errors="raise"),
            "execution_time_s": pd.to_numeric(raw[time], errors="raise"),
            "estimated_energy_kj": pd.to_numeric(raw[energy], errors="raise"),
        }
    )
    if str(energy).strip().lower().endswith("_j"):
        runs["estimated_energy_kj"] /= 1000.0
    return runs


def validate(runs: pd.DataFrame) -> None:
    if len(runs) != 75:
        raise ValueError(f"Expected 75 primary runs, found {len(runs)}")
    if set(runs["algorithm"]) != set(ALGORITHMS):
        raise ValueError(f"Unexpected algorithms: {sorted(runs['algorithm'].unique())}")
    if not np.allclose(sorted(runs["join_ratio"].unique()), JOIN_RATIOS):
        raise ValueError(f"Unexpected join ratios: {sorted(runs['join_ratio'].unique())}")
    counts = runs.groupby(["algorithm", "join_ratio"]).size()
    if len(counts) != 15 or not counts.eq(5).all():
        raise ValueError("Expected five repetitions in every configuration")
    if runs[METRICS].isna().any().any():
        raise ValueError("Missing values found in analysis metrics")


def descriptive_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algorithm, q), group in runs.groupby(["algorithm", "join_ratio"], sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            q1, q3 = np.percentile(values, [25, 75])
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            rows.append(
                {
                    "algorithm": algorithm,
                    "join_ratio": q,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "median": float(np.median(values)),
                    "standard_deviation": sd,
                    "minimum": float(np.min(values)),
                    "q1": float(q1),
                    "q3": float(q3),
                    "maximum": float(np.max(values)),
                    "iqr": float(q3 - q1),
                    "coefficient_of_variation_percent": (
                        100.0 * sd / abs(mean) if mean != 0 else np.nan
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["algorithm"] = pd.Categorical(
        result["algorithm"], categories=ALGORITHMS, ordered=True
    )
    return result.sort_values(["metric", "algorithm", "join_ratio"]).reset_index(drop=True)


def flag_outliers(runs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (algorithm, q), group in runs.groupby(["algorithm", "join_ratio"], sort=False):
        for metric in METRICS:
            q1, q3 = np.percentile(group[metric], [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            for row in group.itertuples(index=False):
                value = float(getattr(row, metric))
                is_outlier = bool(value < lower or value > upper)
                if is_outlier:
                    direction = "below_lower_fence" if value < lower else "above_upper_fence"
                    records.append(
                        {
                            "algorithm": algorithm,
                            "join_ratio": q,
                            "rep": row.rep,
                            "metric": metric,
                            "value": value,
                            "q1": q1,
                            "q3": q3,
                            "iqr": iqr,
                            "lower_fence": lower,
                            "upper_fence": upper,
                            "direction": direction,
                        }
                    )
    columns = [
        "algorithm", "join_ratio", "rep", "metric", "value", "q1", "q3",
        "iqr", "lower_fence", "upper_fence", "direction",
    ]
    return pd.DataFrame(records, columns=columns)


def stability_ranking(stats: pd.DataFrame) -> pd.DataFrame:
    ranked = stats.copy()
    ranked["cv_rank_within_metric"] = ranked.groupby("metric", observed=True)[
        "coefficient_of_variation_percent"
    ].rank(method="min", ascending=True)
    return ranked.sort_values(["metric", "cv_rank_within_metric"]).reset_index(drop=True)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_cv(stats: pd.DataFrame, metric: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    subset = stats[stats["metric"] == metric]
    for algorithm in ALGORITHMS:
        part = subset[subset["algorithm"] == algorithm].sort_values("join_ratio")
        ax.plot(
            part["join_ratio"],
            part["coefficient_of_variation_percent"],
            marker="o",
            linewidth=2,
            label=algorithm,
        )
    ax.set_xlabel("Join ratio")
    ax.set_ylabel("Coefficient of variation (%)")
    ax.set_title(f"Stability across repetitions: {METRIC_LABELS[metric]}")
    ax.set_xticks(JOIN_RATIOS)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    save_figure(fig, output_dir / f"cv_{metric}")


def write_report(
    stats: pd.DataFrame, outliers: pd.DataFrame, output_dir: Path
) -> None:
    lines = [
        "REPETITION STABILITY ANALYSIS",
        "Primary executions: 75 (five repetitions per configuration)",
        "Metrics: best accuracy, execution time, and estimated energy",
        "Potential outliers: Tukey rule (values outside Q1 - 1.5*IQR or Q3 + 1.5*IQR)",
        "No execution is automatically excluded",
        "",
        "COEFFICIENT OF VARIATION BY CONFIGURATION",
    ]
    for metric in METRICS:
        lines.append("")
        lines.append(metric)
        subset = stats[stats["metric"] == metric]
        for row in subset.itertuples(index=False):
            lines.append(
                f"{row.algorithm} | q={row.join_ratio:.2f} | "
                f"mean={row.mean:.9f} | median={row.median:.9f} | "
                f"sd={row.standard_deviation:.9f} | iqr={row.iqr:.9f} | "
                f"cv_percent={row.coefficient_of_variation_percent:.6f}"
            )
    lines.extend(["", "POTENTIAL OUTLIERS"])
    if outliers.empty:
        lines.append("None flagged by the Tukey rule")
    else:
        for row in outliers.sort_values(["metric", "algorithm", "join_ratio", "rep"]).itertuples(index=False):
            lines.append(
                f"{row.metric} | {row.algorithm} | q={row.join_ratio:.2f} | "
                f"rep={row.rep} | value={row.value:.9f} | {row.direction}"
            )
    lines.extend(
        [
            "",
            "CAUTION",
            "With only five repetitions, quartiles and Tukey fences are sensitive to individual observations.",
            "Flags are diagnostic indications for traceability, not grounds for automatic exclusion.",
            "The coefficient of variation should only be compared among measurements on compatible scales.",
        ]
    )
    (output_dir / "stability_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    validate(runs)
    stats = descriptive_statistics(runs)
    outliers = flag_outliers(runs)
    ranking = stability_ranking(stats)

    stats.to_csv(args.output_dir / "stability_descriptive_statistics.csv", index=False)
    outliers.to_csv(args.output_dir / "potential_outliers.csv", index=False)
    ranking.to_csv(args.output_dir / "stability_cv_ranking.csv", index=False)
    configure_plots()
    for metric in METRICS:
        plot_cv(stats, metric, args.output_dir)
    write_report(stats, outliers, args.output_dir)

    print(f"input_runs: {len(runs)}")
    print(f"descriptive_rows: {len(stats)}")
    print(f"potential_outlier_flags: {len(outliers)}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
