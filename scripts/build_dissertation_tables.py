#!/usr/bin/env python3
"""Build compact, auditable dissertation tables from existing analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ALGORITHMS = ["FedAvg", "FedProx", "SCAFFOLD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis file not found: {path}")
    return pd.read_csv(path)


def write_csv_and_tex(
    full: pd.DataFrame,
    latex: pd.DataFrame,
    stem: str,
    output_dir: Path,
    column_format: str | None = None,
) -> None:
    full.to_csv(output_dir / f"{stem}.csv", index=False)
    text = latex.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        column_format=column_format,
    )
    (output_dir / f"{stem}.tex").write_text(text, encoding="utf-8")


def yn(value: object) -> str:
    return "Sim" if bool(value) else "Não"


def configuration_table(root: Path, output_dir: Path) -> pd.DataFrame:
    pareto = read_required(root / "pareto_tradeoff" / "pareto_configurations.csv")
    convergence = read_required(
        root / "convergence" / "convergence_configuration_summary.csv"
    )
    stability = read_required(
        root / "stability" / "stability_descriptive_statistics.csv"
    )
    cv = stability.pivot(
        index=["algorithm", "join_ratio"],
        columns="metric",
        values="coefficient_of_variation_percent",
    ).reset_index()
    cv = cv.rename(
        columns={
            "best_accuracy": "accuracy_cv_percent",
            "execution_time_s": "time_cv_percent",
            "estimated_energy_kj": "energy_cv_percent",
        }
    )
    conv_cols = [
        "algorithm",
        "join_ratio",
        "normalized_aulc_mean",
        "first_round_relative_95_median",
        "first_round_absolute_median",
        "sustained_round_absolute_median",
        "sustained_round_absolute_reached_n",
    ]
    table = pareto.merge(
        convergence[conv_cols], on=["algorithm", "join_ratio"], validate="one_to_one"
    ).merge(cv, on=["algorithm", "join_ratio"], validate="one_to_one")
    table["algorithm"] = pd.Categorical(
        table["algorithm"], categories=ALGORITHMS, ordered=True
    )
    table = table.sort_values(["algorithm", "join_ratio"]).reset_index(drop=True)

    latex = pd.DataFrame(
        {
            "Algoritmo": table["algorithm"].astype(str),
            "$q$": table["join_ratio"].map(lambda x: f"{x:.2f}"),
            "Accuracy": table.apply(
                lambda r: f"{r.accuracy_mean:.4f} $\\pm$ {r.accuracy_std:.4f}", axis=1
            ),
            "Tempo (s)": table["time_median_s"].map(lambda x: f"{x:.2f}"),
            "Energia (kJ)": table["energy_median_kj"].map(lambda x: f"{x:.2f}"),
            "AULC": table["normalized_aulc_mean"].map(lambda x: f"{x:.4f}"),
            "$r_{95}$": table["first_round_relative_95_median"].map(
                lambda x: f"{x:.0f}"
            ),
            "Pareto 3D": table["pareto_accuracy_time_energy_global"].map(yn),
        }
    )
    write_csv_and_tex(table, latex, "table_configuration_overview", output_dir)
    return table


def global_effect_table(root: Path, output_dir: Path) -> None:
    table = read_required(
        root / "join_ratio_effect" / "kruskal_within_algorithm.csv"
    )
    metric_names = {
        "best_accuracy": "Accuracy",
        "execution_time_s": "Tempo",
        "estimated_energy_kj": "Energia",
    }
    latex = pd.DataFrame(
        {
            "Métrica": table["metric"].map(metric_names),
            "Algoritmo": table["algorithm"],
            "$H$": table["kruskal_h"].map(lambda x: f"{x:.3f}"),
            "$p$": table["p_value"].map(lambda x: f"{x:.6f}"),
            "$\\epsilon^2$": table["epsilon_squared"].map(lambda x: f"{x:.3f}"),
        }
    )
    write_csv_and_tex(table, latex, "table_join_ratio_global_effect", output_dir)


def marginal_table(root: Path, output_dir: Path) -> None:
    table = read_required(root / "marginal_returns" / "marginal_returns.csv")
    latex = pd.DataFrame(
        {
            "Algoritmo": table["algorithm"],
            "Transição": table["transition"],
            "$\\Delta$acc. (pp)": table["delta_accuracy_percentage_points"].map(
                lambda x: f"{x:+.3f}"
            ),
            "$\\Delta$t (s)": table["delta_time_s"].map(lambda x: f"{x:+.1f}"),
            "$\\Delta$E (kJ)": table["delta_energy_kj"].map(lambda x: f"{x:+.2f}"),
            "pp/1000 s": table["accuracy_pp_per_1000_additional_s"].map(
                lambda x: f"{x:+.3f}"
            ),
            "pp/kJ": table["accuracy_pp_per_additional_kj"].map(
                lambda x: f"{x:+.4f}"
            ),
        }
    )
    write_csv_and_tex(table, latex, "table_marginal_returns", output_dir)


def recommendations_table(root: Path, output_dir: Path) -> None:
    table = read_required(root / "pareto_tradeoff" / "recommendations.csv")
    category_names = {
        "maximum_quality": "Máxima qualidade",
        "minimum_time": "Menor tempo",
        "maximum_energy_efficiency": "Maior eficiência energética",
        "balanced_accuracy_time": "Compromisso accuracy--tempo",
        "balanced_accuracy_energy": "Compromisso accuracy--energia",
        "balanced_accuracy_time_energy": "Compromisso tridimensional",
    }
    latex = pd.DataFrame(
        {
            "Critério": table["category"].map(category_names).fillna(table["category"]),
            "Configuração": table["configuration"].str.replace("_", "\\_", regex=False),
            "Accuracy": table["accuracy_mean"].map(lambda x: f"{x:.4f}"),
            "Tempo (s)": table["time_median_s"].map(lambda x: f"{x:.2f}"),
            "Energia (kJ)": table["energy_median_kj"].map(lambda x: f"{x:.2f}"),
        }
    )
    write_csv_and_tex(table, latex, "table_recommendations", output_dir)


def knees_table(root: Path, output_dir: Path) -> None:
    table = read_required(root / "marginal_returns" / "knee_candidates.csv")
    latex = pd.DataFrame(
        {
            "Algoritmo": table["algorithm"],
            "Custo": table["cost_dimension"].map(
                {"time": "Tempo", "energy": "Energia"}
            ),
            "$q$ candidato": table["candidate_join_ratio"].map(lambda x: f"{x:.2f}"),
            "Distância": table["normalized_distance_to_chord"].map(
                lambda x: f"{x:.3f}"
            ),
        }
    )
    write_csv_and_tex(table, latex, "table_exploratory_knees", output_dir)


def workload_table(root: Path, output_dir: Path) -> None:
    workload = read_required(
        root / "system_workload" / "system_workload_by_configuration.csv"
    )
    comm = read_required(root / "system_workload" / "communication_scenarios.csv")
    base = workload[workload["algorithm"] == "FedAvg"].copy()
    base = base[
        [
            "join_ratio",
            "selected_clients_per_cycle",
            "nominal_client_participations_50_rounds",
            "actual_client_participations_51_cycles",
            "actual_local_training_sample_presentations",
            "evaluation_sample_presentations_all_clients",
        ]
    ]
    selected = comm[comm["scenario"] == "selected_only_distributed_scenario"]
    payload = selected.pivot(
        index="join_ratio", columns="algorithm", values="logical_payload_gb_51_cycles"
    ).reset_index()
    payload = payload.rename(
        columns={
            "FedAvg": "fedavg_payload_gb",
            "FedProx": "fedprox_payload_gb",
            "SCAFFOLD": "scaffold_payload_gb",
        }
    )
    table = base.merge(payload, on="join_ratio", validate="one_to_one").sort_values(
        "join_ratio"
    )
    latex = pd.DataFrame(
        {
            "$q$": table["join_ratio"].map(lambda x: f"{x:.2f}"),
            "Clientes/ciclo": table["selected_clients_per_cycle"],
            "Participações (51)": table["actual_client_participations_51_cycles"],
            "Amostras locais": table[
                "actual_local_training_sample_presentations"
            ].map(lambda x: f"{x / 1e6:.3f} M"),
            "FedAvg/Prox (GB)": table["fedavg_payload_gb"].map(lambda x: f"{x:.2f}"),
            "SCAFFOLD (GB)": table["scaffold_payload_gb"].map(lambda x: f"{x:.2f}"),
        }
    )
    write_csv_and_tex(table, latex, "table_system_workload", output_dir)


def outlier_summary(root: Path, output_dir: Path) -> None:
    outliers = read_required(root / "stability" / "potential_outliers.csv")
    if outliers.empty:
        summary = pd.DataFrame(columns=["metric", "algorithm", "flag_count"])
    else:
        summary = (
            outliers.groupby(["metric", "algorithm"], as_index=False)
            .size()
            .rename(columns={"size": "flag_count"})
        )
    latex = summary.rename(
        columns={"metric": "Métrica", "algorithm": "Algoritmo", "flag_count": "Flags"}
    )
    write_csv_and_tex(summary, latex, "table_outlier_flag_summary", output_dir)


def write_manifest(output_dir: Path, configuration_rows: int) -> None:
    descriptions = [
        ("table_configuration_overview", "Síntese principal das 15 configurações"),
        ("table_join_ratio_global_effect", "Efeito global do join ratio por algoritmo"),
        ("table_marginal_returns", "Ganhos e custos marginais consecutivos"),
        ("table_recommendations", "Configurações recomendadas por critério"),
        ("table_exploratory_knees", "Joelhos exploratórios das curvas"),
        ("table_system_workload", "Carga computacional e payload lógico"),
        ("table_outlier_flag_summary", "Resumo de flags diagnósticos de outliers"),
    ]
    manifest = pd.DataFrame(descriptions, columns=["table", "purpose"])
    manifest["csv_file"] = manifest["table"] + ".csv"
    manifest["latex_file"] = manifest["table"] + ".tex"
    manifest.to_csv(output_dir / "tables_manifest.csv", index=False)
    lines = [
        "DISSERTATION TABLE CONSOLIDATION",
        f"Configurations consolidated: {configuration_rows}",
        f"Tables generated: {len(manifest)}",
        "Each table is available as an audit CSV and a LaTeX tabular fragment.",
        "No experiment or inferential test is executed by this consolidation step.",
        "",
        "USAGE NOTES",
        "Accuracy overview uses the precision-preserving article metric.",
        "Convergence metrics use round-wise logged accuracy and therefore its logged precision.",
        "Time and energy are summarized by medians; accuracy is summarized by mean and SD.",
        "AULC is normalized over rounds 0..50.",
        "The absolute convergence threshold is 0.65.",
        "Communication is a logical FP32 model-payload estimate, not measured network traffic.",
        "Knee points and Tukey outliers are exploratory diagnostics, not exclusion rules.",
    ]
    (output_dir / "consolidation_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview = configuration_table(args.analysis_root, args.output_dir)
    global_effect_table(args.analysis_root, args.output_dir)
    marginal_table(args.analysis_root, args.output_dir)
    recommendations_table(args.analysis_root, args.output_dir)
    knees_table(args.analysis_root, args.output_dir)
    workload_table(args.analysis_root, args.output_dir)
    outlier_summary(args.analysis_root, args.output_dir)
    write_manifest(args.output_dir, len(overview))
    print(f"configuration_rows: {len(overview)}")
    print("tables_generated: 7")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
