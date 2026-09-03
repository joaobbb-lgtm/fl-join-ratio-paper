#!/usr/bin/env python3
"""Analytical characterization of the workload produced by the LANC experiment.

This script does not run new FL experiments.  It combines the observed run-level
time/energy measurements with workload counts derived from the experiment and
the inspected PFLlib implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = ["FedAvg", "FedProx", "SCAFFOLD"]
JOIN_RATIOS = [0.10, 0.25, 0.50, 0.75, 1.00]

# Fixed properties of the validated FashionMNIST experiment.
N_CLIENTS = 100
NOMINAL_ROUNDS = 50
ACTUAL_TRAIN_CYCLES = 51  # range(global_rounds + 1) in the inspected server loop
EVALUATIONS = 51          # initial evaluation plus evaluations after updates 1..50
LOCAL_EPOCHS = 5
TRAIN_SAMPLES_PER_CLIENT = 525
TEST_SAMPLES_PER_CLIENT = 175
BATCH_SIZE = 10
TRAIN_BATCHES_PER_EPOCH = 52  # drop_last=True: floor(525 / 10)
TEST_BATCHES_PER_EVAL = 18    # drop_last=False: ceil(175 / 10)
MODEL_PARAMETERS = 582_026
BYTES_PER_PARAMETER = 4       # FP32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f"Missing required column. Expected one of {candidates}; "
        f"found {list(df.columns)}"
    )


def load_runs(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    columns = {
        "algorithm": find_column(raw, ["algorithm", "algo"]),
        "join_ratio": find_column(raw, ["join_ratio", "join ratio", "ratio"]),
        "execution_time_s": find_column(
            raw,
            [
                "execution_time_s",
                "execution_time",
                "total_time_reported_s",
                "total_time_s",
                "time_s",
            ]
        ),
        "estimated_energy_kj": find_column(
            raw,
            [
                "estimated_energy_kj",
                "energy_kj",
                "estimated_energy",
                "estimated_energy_j",
            ]
        ),
    }
    runs = raw[[columns[k] for k in columns]].rename(
        columns={v: k for k, v in columns.items()}
    )
    runs["algorithm"] = runs["algorithm"].astype(str).str.strip()

    for col in ["join_ratio", "execution_time_s", "estimated_energy_kj"]:
        runs[col] = pd.to_numeric(runs[col], errors="raise")

    source_energy_column = str(columns["estimated_energy_kj"]).strip().lower()
    if source_energy_column.endswith("_j"):
        runs["estimated_energy_kj"] = runs["estimated_energy_kj"] / 1000.0

    return runs


def validate_design(runs: pd.DataFrame) -> None:
    if len(runs) != 75:
        raise ValueError(f"Expected 75 primary runs, found {len(runs)}")
    unknown = sorted(set(runs["algorithm"]) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"Unexpected algorithms: {unknown}")
    observed_q = sorted(runs["join_ratio"].round(8).unique().tolist())
    if not np.allclose(observed_q, JOIN_RATIOS):
        raise ValueError(f"Unexpected join ratios: {observed_q}")
    counts = runs.groupby(["algorithm", "join_ratio"]).size()
    if len(counts) != 15 or not (counts == 5).all():
        raise ValueError("Expected exactly five runs in each of 15 configurations")


def assumptions_table() -> pd.DataFrame:
    rows = [
        ("clients_total", N_CLIENTS, "clients", "experiment configuration"),
        ("nominal_global_rounds", NOMINAL_ROUNDS, "rounds", "experiment configuration"),
        ("actual_training_cycles", ACTUAL_TRAIN_CYCLES, "cycles", "inspected PFLlib loop"),
        ("global_evaluations", EVALUATIONS, "evaluations", "rounds 0..50 in logs"),
        ("local_epochs", LOCAL_EPOCHS, "epochs", "experiment configuration"),
        ("train_samples_per_client", TRAIN_SAMPLES_PER_CLIENT, "samples", "fixed partition"),
        ("test_samples_per_client", TEST_SAMPLES_PER_CLIENT, "samples", "fixed partition"),
        ("batch_size", BATCH_SIZE, "samples", "experiment configuration"),
        ("train_batches_per_epoch", TRAIN_BATCHES_PER_EPOCH, "batches", "drop_last=True"),
        ("test_batches_per_evaluation", TEST_BATCHES_PER_EVAL, "batches", "drop_last=False"),
        ("model_trainable_parameters", MODEL_PARAMETERS, "parameters", "FedAvgCNN architecture"),
        ("parameter_precision", 32, "bits", "FP32 assumption"),
        ("model_vector_size", MODEL_PARAMETERS * BYTES_PER_PARAMETER, "bytes", "parameters x 4"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "unit", "basis"])


def build_workload(runs: pd.DataFrame) -> pd.DataFrame:
    observed = (
        runs.groupby(["algorithm", "join_ratio"], as_index=False)
        .agg(
            repetitions=("execution_time_s", "size"),
            median_execution_time_s=("execution_time_s", "median"),
            median_estimated_energy_kj=("estimated_energy_kj", "median"),
        )
    )
    rows = []
    train_batches_per_participation = LOCAL_EPOCHS * TRAIN_BATCHES_PER_EPOCH
    train_sample_presentations_per_participation = train_batches_per_participation * BATCH_SIZE
    eval_train_batches = N_CLIENTS * TRAIN_BATCHES_PER_EPOCH
    eval_test_batches = N_CLIENTS * TEST_BATCHES_PER_EVAL
    eval_train_sample_presentations = N_CLIENTS * TRAIN_BATCHES_PER_EPOCH * BATCH_SIZE
    eval_test_samples = N_CLIENTS * TEST_SAMPLES_PER_CLIENT

    for row in observed.itertuples(index=False):
        selected = int(round(N_CLIENTS * row.join_ratio))
        nominal_participations = selected * NOMINAL_ROUNDS
        actual_participations = selected * ACTUAL_TRAIN_CYCLES
        training_batches = actual_participations * train_batches_per_participation
        training_samples = actual_participations * train_sample_presentations_per_participation
        rows.append(
            {
                "algorithm": row.algorithm,
                "join_ratio": row.join_ratio,
                "selected_clients_per_cycle": selected,
                "nominal_client_participations_50_rounds": nominal_participations,
                "actual_client_participations_51_cycles": actual_participations,
                "actual_local_training_batches": training_batches,
                "actual_local_training_sample_presentations": training_samples,
                "evaluation_train_batches_all_clients": EVALUATIONS * eval_train_batches,
                "evaluation_test_batches_all_clients": EVALUATIONS * eval_test_batches,
                "evaluation_sample_presentations_all_clients": EVALUATIONS
                * (eval_train_sample_presentations + eval_test_samples),
                "median_execution_time_s": row.median_execution_time_s,
                "median_estimated_energy_kj": row.median_estimated_energy_kj,
                "observed_seconds_per_client_participation": row.median_execution_time_s
                / actual_participations,
                "observed_kj_per_client_participation": row.median_estimated_energy_kj
                / actual_participations,
            }
        )
    return pd.DataFrame(rows)


def communication_table(workload: pd.DataFrame) -> pd.DataFrame:
    vector_bytes = MODEL_PARAMETERS * BYTES_PER_PARAMETER
    rows = []
    for r in workload.itertuples(index=False):
        q_clients = r.selected_clients_per_cycle
        if r.algorithm == "SCAFFOLD":
            down_all_vectors = 2 * N_CLIENTS
            up_selected_vectors = 2 * q_clients
            selected_only_vectors = 4 * q_clients
        else:
            down_all_vectors = N_CLIENTS
            up_selected_vectors = q_clients
            selected_only_vectors = 2 * q_clients

        scenarios = [
            (
                "pflib_logical_copy_pattern",
                down_all_vectors,
                up_selected_vectors,
                "all-client downlink pattern observed in code; logical payload, not network traffic",
            ),
            (
                "selected_only_distributed_scenario",
                selected_only_vectors // 2,
                selected_only_vectors - selected_only_vectors // 2,
                "hypothetical implementation communicating only with selected clients",
            ),
        ]
        for scenario, down, up, note in scenarios:
            total_vectors = (down + up) * ACTUAL_TRAIN_CYCLES
            rows.append(
                {
                    "algorithm": r.algorithm,
                    "join_ratio": r.join_ratio,
                    "scenario": scenario,
                    "downlink_vectors_per_cycle": down,
                    "uplink_vectors_per_cycle": up,
                    "total_vectors_51_cycles": total_vectors,
                    "logical_payload_bytes_51_cycles": total_vectors * vector_bytes,
                    "logical_payload_gb_51_cycles": total_vectors * vector_bytes / 1e9,
                    "interpretation": note,
                }
            )
    return pd.DataFrame(rows)


def set_plot_style() -> None:
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


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_workload(workload: pd.DataFrame, outdir: Path) -> None:
    data = workload[workload["algorithm"] == "FedAvg"].sort_values("join_ratio")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        data["join_ratio"],
        data["actual_local_training_sample_presentations"] / 1e6,
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("Join ratio")
    ax.set_ylabel("Local training sample presentations (millions)")
    ax.set_title("Analytical local-training workload per execution")
    ax.set_xticks(JOIN_RATIOS)
    ax.set_ylim(bottom=0)
    save_figure(fig, outdir / "local_training_workload")


def plot_communication(comm: pd.DataFrame, outdir: Path) -> None:
    data = comm[comm["scenario"] == "selected_only_distributed_scenario"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for algorithm in ALGORITHMS:
        part = data[data["algorithm"] == algorithm].sort_values("join_ratio")
        ax.plot(
            part["join_ratio"],
            part["logical_payload_gb_51_cycles"],
            marker="o",
            linewidth=2,
            label=algorithm,
        )
    ax.set_xlabel("Join ratio")
    ax.set_ylabel("Logical model payload (GB)")
    ax.set_title("Selected-client communication scenario (51 cycles, FP32)")
    ax.set_xticks(JOIN_RATIOS)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    save_figure(fig, outdir / "logical_communication_selected_only")


def write_report(workload: pd.DataFrame, comm: pd.DataFrame, outdir: Path) -> None:
    vector_mb = MODEL_PARAMETERS * BYTES_PER_PARAMETER / 1e6
    eval_samples = int(workload["evaluation_sample_presentations_all_clients"].iloc[0])
    lines = [
        "SYSTEM WORKLOAD CHARACTERIZATION",
        "Method: analytical counts derived from the validated experiment and inspected PFLlib code",
        "No new federated-learning execution was performed",
        "",
        "IMPLEMENTATION FACTS",
        f"Model trainable parameters: {MODEL_PARAMETERS}",
        f"One FP32 model vector: {vector_mb:.6f} MB (decimal)",
        f"Nominal rounds: {NOMINAL_ROUNDS}",
        f"Actual training cycles in the inspected range(global_rounds + 1) loop: {ACTUAL_TRAIN_CYCLES}",
        f"Logged all-client evaluations: {EVALUATIONS}",
        f"Fixed evaluation sample presentations per execution: {eval_samples}",
        "",
        "WORKLOAD BY JOIN RATIO (same counts for all algorithms)",
    ]
    base = workload[workload["algorithm"] == "FedAvg"].sort_values("join_ratio")
    for r in base.itertuples(index=False):
        lines.append(
            f"q={r.join_ratio:.2f} | selected/cycle={r.selected_clients_per_cycle} | "
            f"participations_50={r.nominal_client_participations_50_rounds} | "
            f"participations_51={r.actual_client_participations_51_cycles} | "
            f"training_batches={r.actual_local_training_batches} | "
            f"training_sample_presentations={r.actual_local_training_sample_presentations}"
        )
    lines.extend(["", "SELECTED-ONLY LOGICAL COMMUNICATION (51 cycles, FP32)"])
    selected = comm[comm["scenario"] == "selected_only_distributed_scenario"]
    for r in selected.sort_values(["algorithm", "join_ratio"]).itertuples(index=False):
        lines.append(
            f"{r.algorithm} | q={r.join_ratio:.2f} | "
            f"payload_gb={r.logical_payload_gb_51_cycles:.6f}"
        )
    lines.extend(
        [
            "",
            "CAUTION",
            "Communication values are logical model-payload estimates, not measured network traffic.",
            "The PFLlib experiment is an in-process simulation that copies model parameters in memory.",
            "Its server sends parameters to all 100 client objects each cycle; the selected-only scenario is hypothetical.",
            "SCAFFOLD is represented by two parameter-sized vectors in each direction (model/control updates).",
            "Time and energy per participation are amortized observed ratios; they do not isolate causal client cost",
            "because every execution also contains fixed all-client evaluation and other server overheads.",
        ]
    )
    (outdir / "system_workload_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    validate_design(runs)
    assumptions = assumptions_table()
    workload = build_workload(runs)
    comm = communication_table(workload)

    assumptions.to_csv(args.output_dir / "model_and_dataset_assumptions.csv", index=False)
    workload.to_csv(args.output_dir / "system_workload_by_configuration.csv", index=False)
    comm.to_csv(args.output_dir / "communication_scenarios.csv", index=False)
    set_plot_style()
    plot_training_workload(workload, args.output_dir)
    plot_communication(comm, args.output_dir)
    write_report(workload, comm, args.output_dir)

    print(f"input_runs: {len(runs)}")
    print(f"configuration_rows: {len(workload)}")
    print(f"communication_rows: {len(comm)}")
    print(f"model_parameters: {MODEL_PARAMETERS}")
    print(f"actual_training_cycles: {ACTUAL_TRAIN_CYCLES}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
