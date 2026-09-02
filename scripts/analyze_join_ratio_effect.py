"""Analyze the effect of join ratio within each FL algorithm.

Input:  analysis_outputs/dissertation_runs.csv
Output: CSV tables, an audit summary, and reproducible PDF/PNG boxplots.

The primary inferential analysis treats join-ratio groups as independent because
the available metadata do not establish paired seeds across join ratios.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ALGORITHMS = ("FedAvg", "FedProx", "SCAFFOLD")
JOIN_RATIOS = (0.10, 0.25, 0.50, 0.75, 1.00)
METRICS = {
    "best_accuracy": ("best_accuracy_reported", "Best test accuracy"),
    "execution_time_s": ("total_time_reported_s", "Execution time (s)"),
    "estimated_energy_kj": ("estimated_energy_kj", "Estimated energy (kJ)"),
}
COLORS = {"FedAvg": "#185FA5", "FedProx": "#0F6E56", "SCAFFOLD": "#993C1D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "algorithm", "join_ratio", "rep", "planned_run",
        "best_accuracy_reported", "total_time_reported_s", "estimated_energy_j",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    if len(frame) != 75:
        raise ValueError(f"Expected 75 planned runs, found {len(frame)}")
    counts = frame.groupby(["algorithm", "join_ratio"]).size()
    if len(counts) != 15 or not (counts == 5).all():
        raise ValueError("Expected exactly five runs in each of the 15 configurations")
    if set(frame["algorithm"]) != set(ALGORITHMS):
        raise ValueError("Unexpected algorithm set")
    observed_jr = set(np.round(frame["join_ratio"].astype(float), 2))
    if observed_jr != set(JOIN_RATIOS):
        raise ValueError("Unexpected join-ratio set")


def tukey_outlier_count(values: pd.Series) -> int:
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())


def descriptive_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, (column, _) in METRICS.items():
        for (algorithm, join_ratio), group in frame.groupby(["algorithm", "join_ratio"], sort=True):
            values = group[column].astype(float)
            mean = values.mean()
            sd = values.std(ddof=1)
            q1, q3 = values.quantile([0.25, 0.75])
            rows.append({
                "metric": metric,
                "algorithm": algorithm,
                "join_ratio": join_ratio,
                "n": len(values),
                "mean": mean,
                "median": values.median(),
                "std": sd,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "minimum": values.min(),
                "maximum": values.max(),
                "cv_percent": 100.0 * sd / mean if mean != 0 else np.nan,
                "tukey_outliers": tukey_outlier_count(values),
            })
    return pd.DataFrame(rows)


def epsilon_squared_kruskal(h_stat: float, total_n: int, groups: int) -> float:
    # Bias-corrected epsilon-squared estimate for Kruskal-Wallis.
    return max(0.0, (h_stat - groups + 1) / (total_n - groups))


def kruskal_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, (column, _) in METRICS.items():
        for algorithm in ALGORITHMS:
            subset = frame[frame["algorithm"] == algorithm]
            groups = [
                subset[np.isclose(subset["join_ratio"], jr)][column].astype(float).to_numpy()
                for jr in JOIN_RATIOS
            ]
            h_stat, p_value = stats.kruskal(*groups)
            rows.append({
                "metric": metric,
                "algorithm": algorithm,
                "n_total": sum(map(len, groups)),
                "groups": len(groups),
                "kruskal_h": h_stat,
                "p_value": p_value,
                "epsilon_squared": epsilon_squared_kruskal(h_stat, sum(map(len, groups)), len(groups)),
                "significant_0_05": bool(p_value < 0.05),
            })
    return pd.DataFrame(rows)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    comparisons = np.sign(x[:, None] - y[None, :])
    return float(comparisons.sum() / comparisons.size)


def cliffs_magnitude(delta: float) -> str:
    value = abs(delta)
    if value < 0.147:
        return "negligible"
    if value < 0.330:
        return "small"
    if value < 0.474:
        return "medium"
    return "large"


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def pairwise_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, (column, _) in METRICS.items():
        for algorithm in ALGORITHMS:
            subset = frame[frame["algorithm"] == algorithm]
            local_rows, p_values = [], []
            for jr_a, jr_b in itertools.combinations(JOIN_RATIOS, 2):
                x = subset[np.isclose(subset["join_ratio"], jr_a)][column].astype(float).to_numpy()
                y = subset[np.isclose(subset["join_ratio"], jr_b)][column].astype(float).to_numpy()
                u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided", method="exact")
                delta = cliffs_delta(y, x)  # positive means the higher join ratio has larger values
                local_rows.append({
                    "metric": metric,
                    "algorithm": algorithm,
                    "join_ratio_a": jr_a,
                    "join_ratio_b": jr_b,
                    "mann_whitney_u": u_stat,
                    "p_value_raw": p_value,
                    "cliffs_delta_b_minus_a": delta,
                    "cliffs_magnitude": cliffs_magnitude(delta),
                })
                p_values.append(p_value)
            for row, adjusted in zip(local_rows, holm_adjust(p_values)):
                row["p_value_holm"] = adjusted
                row["significant_holm_0_05"] = bool(adjusted < 0.05)
                rows.append(row)
    return pd.DataFrame(rows)


def plot_boxplots(frame: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(20260902)
    for metric, (column, ylabel) in METRICS.items():
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=False)
        for axis, algorithm in zip(axes, ALGORITHMS):
            subset = frame[frame["algorithm"] == algorithm]
            groups = [
                subset[np.isclose(subset["join_ratio"], jr)][column].astype(float).to_numpy()
                for jr in JOIN_RATIOS
            ]
            axis.boxplot(groups, widths=0.55, showfliers=False, patch_artist=True,
                         boxprops={"facecolor": COLORS[algorithm], "alpha": 0.25},
                         medianprops={"color": COLORS[algorithm], "linewidth": 1.8})
            for position, values in enumerate(groups, start=1):
                jitter = rng.uniform(-0.08, 0.08, size=len(values))
                axis.scatter(position + jitter, values, s=22, color=COLORS[algorithm],
                             edgecolor="white", linewidth=0.4, zorder=3)
            axis.set_title(algorithm)
            axis.set_xticks(range(1, 6), [f"{jr:.2f}" for jr in JOIN_RATIOS], rotation=30)
            axis.set_xlabel("Join ratio")
            axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"{ylabel} by join ratio (n=5 per configuration)", fontsize=12)
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(output_dir / f"boxplot_{metric}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_summary(kruskal: pd.DataFrame, pairwise: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "JOIN-RATIO EFFECT ANALYSIS",
        "Primary runs: 75 (five per algorithm/join-ratio configuration)",
        "Design used for inference: independent groups",
        "Global test: Kruskal-Wallis within each algorithm",
        "Post hoc: exact two-sided Mann-Whitney U with Holm correction",
        "Effect sizes: Kruskal epsilon-squared and pairwise Cliff's delta",
        "",
        "GLOBAL RESULTS",
    ]
    for row in kruskal.itertuples(index=False):
        lines.append(
            f"{row.metric} | {row.algorithm} | H={row.kruskal_h:.6f} | "
            f"p={row.p_value:.8f} | epsilon2={row.epsilon_squared:.6f}"
        )
    lines.extend(["", "HOLM-SIGNIFICANT PAIRWISE COMPARISONS"])
    significant = pairwise[pairwise["significant_holm_0_05"]]
    if significant.empty:
        lines.append("None")
    else:
        for row in significant.itertuples(index=False):
            lines.append(
                f"{row.metric} | {row.algorithm} | {row.join_ratio_a:.2f} vs "
                f"{row.join_ratio_b:.2f} | p_holm={row.p_value_holm:.8f} | "
                f"delta={row.cliffs_delta_b_minus_a:.3f} ({row.cliffs_magnitude})"
            )
    (output_dir / "join_ratio_effect_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.runs)
    validate_input(frame)
    frame["estimated_energy_kj"] = frame["estimated_energy_j"] / 1000.0

    descriptive = descriptive_table(frame)
    kruskal = kruskal_table(frame)
    pairwise = pairwise_table(frame)

    descriptive.to_csv(args.output_dir / "descriptive_statistics.csv", index=False)
    kruskal.to_csv(args.output_dir / "kruskal_within_algorithm.csv", index=False)
    pairwise.to_csv(args.output_dir / "mannwhitney_holm_cliffs_delta.csv", index=False)
    plot_boxplots(frame, args.output_dir)
    write_summary(kruskal, pairwise, args.output_dir)

    print(f"input_runs: {len(frame)}")
    print(f"descriptive_rows: {len(descriptive)}")
    print(f"global_tests: {len(kruskal)}")
    print(f"pairwise_tests: {len(pairwise)}")
    print(f"holm_significant_pairwise: {int(pairwise['significant_holm_0_05'].sum())}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
