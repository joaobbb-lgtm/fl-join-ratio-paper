#!/usr/bin/env python3
"""Build auditable dissertation datasets from the original FashionMNIST logs.

The script never modifies raw logs. It keeps the 75 planned runs (rep1..rep5),
records additional/rerun executions separately, and reproduces the energy proxy
used by the LANC 2026 analysis notebook.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


JR_MAP = {"jr01": 0.10, "jr025": 0.25, "jr05": 0.50, "jr075": 0.75, "jr10": 1.00}
ALGORITHMS = ("FedAvg", "FedProx", "SCAFFOLD")
PLANNED_REPS = {f"rep{i}" for i in range(1, 6)}

# Proxy documented in the article analysis notebook.
CPU_TDP_W = 15.0
GPU_TDP_W = 10.0
RAM_W = 4.0
BASELINE_W = 2.0
CPU_MAX_PCT = 800.0
DT_SECONDS = 1.0

ROUND_BLOCK_RE = re.compile(
    r"(?:\[(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+)?"
    r"[-]+Round number:\s+(?P<round>\d+)[-]+(?P<body>.*?)"
    r"[-]+\s*time cost\s+[-]+\s+(?P<round_time_s>[0-9.]+)",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--article-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_cpu(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "Linux", "Average:")):
            continue
        parts = line.split()
        if len(parts) >= 8:
            try:
                values.append(float(parts[7].replace(",", ".")))
            except ValueError:
                pass
    return np.clip(np.asarray(values, dtype=float) / CPU_MAX_PCT, 0.0, 1.0)


def parse_gpu(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    values = pd.to_numeric(frame["utilization_gpu_pct"], errors="coerce").dropna()
    return np.clip(values.to_numpy(dtype=float) / 100.0, 0.0, 1.0)


def article_energy(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, float | int]:
    if len(cpu) == 0 or len(gpu) == 0:
        raise ValueError("empty CPU or GPU monitoring series")
    n = min(len(cpu), len(gpu))
    power = cpu[:n] * CPU_TDP_W + gpu[:n] * GPU_TDP_W + RAM_W + BASELINE_W
    return {
        "monitor_samples": n,
        "monitor_duration_s": n * DT_SECONDS,
        "cpu_util_fraction_mean": float(cpu[:n].mean()),
        "gpu_util_fraction_mean": float(gpu[:n].mean()),
        "estimated_power_mean_w": float(power.mean()),
        "estimated_power_peak_w": float(power.max()),
        "estimated_energy_j": float(power.sum() * DT_SECONDS),
    }


def extract_scalar(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing {label}")
    return float(match.group(1))


def parse_run(rep_dir: Path, algorithm: str, jr_folder: str, rep: str) -> tuple[dict, list[dict]]:
    raw_path = rep_dir / "raw_log.txt"
    meta_path = rep_dir / "meta_info.txt"
    cpu_path = rep_dir / "cpu_usage.log"
    gpu_path = rep_dir / "gpu_usage.csv"
    for path in (raw_path, meta_path, cpu_path, gpu_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    text = raw_path.read_text(encoding="utf-8", errors="replace")
    metadata = parse_metadata(meta_path)
    energy = article_energy(parse_cpu(cpu_path), parse_gpu(gpu_path))
    rounds = []
    cumulative_time = 0.0
    for match in ROUND_BLOCK_RE.finditer(text):
        values = match.groupdict()
        body = values["body"]
        timestamps = re.findall(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", match.group(0))
        round_time = float(values["round_time_s"])
        cumulative_time += round_time
        rounds.append({
            "algorithm": algorithm,
            "join_ratio": JR_MAP[jr_folder],
            "rep": rep,
            "round": int(values["round"]),
            "evaluation_timestamp": timestamps[0] if timestamps else "",
            "train_loss": extract_scalar(r"Averaged Train Loss:\s+([0-9.]+)", body, "train loss"),
            "test_accuracy": extract_scalar(r"Averaged Test Accuracy:\s+([0-9.]+)", body, "test accuracy"),
            "test_auc": extract_scalar(r"Averaged Test AUC:\s+([0-9.]+)", body, "test AUC"),
            "std_test_accuracy_clients": extract_scalar(r"Std Test Accuracy:\s+([0-9.]+)", body, "std test accuracy"),
            "std_test_auc_clients": extract_scalar(r"Std Test AUC:\s+([0-9.]+)", body, "std test AUC"),
            "iteration_time_s": round_time,
            "cumulative_iteration_time_s": cumulative_time,
        })
    if len(rounds) != 51 or {row["round"] for row in rounds} != set(range(51)):
        raise ValueError(f"expected rounds 0..50, found {len(rounds)} records")

    accuracies = np.asarray([row["test_accuracy"] for row in rounds])
    best_index = int(np.argmax(accuracies))
    run = {
        "algorithm": algorithm,
        "join_ratio": JR_MAP[jr_folder],
        "rep": rep,
        "planned_run": rep in PLANNED_REPS,
        "exit_code": int(metadata["EXIT_CODE"]),
        "global_rounds_parameter": int(metadata["GLOBAL_ROUNDS"]),
        "logged_evaluations": len(rounds),
        "local_epochs": int(metadata["LOCAL_EPOCHS"]),
        "batch_size": int(metadata["BATCH_SIZE"]),
        "learning_rate": float(metadata["LOCAL_LEARNING_RATE"]),
        "best_accuracy_from_rounds": float(accuracies[best_index]),
        "best_accuracy_round": int(rounds[best_index]["round"]),
        "best_accuracy_reported": extract_scalar(r"Best accuracy\.\s*\n([0-9.]+)", text, "best accuracy"),
        "final_accuracy_round_50": float(accuracies[-1]),
        "initial_accuracy_round_0": float(accuracies[0]),
        "total_time_reported_s": extract_scalar(r"Average time cost:\s*([0-9.]+)s", text, "total time"),
        "sum_iteration_time_s": cumulative_time,
        **energy,
    }
    return run, rounds


def compare_with_article(runs: pd.DataFrame, article_path: Path) -> pd.DataFrame:
    article = pd.read_csv(article_path)
    planned = runs[runs["planned_run"]].copy()
    merged = planned.merge(article, on=["algorithm", "join_ratio", "rep"], how="outer", indicator=True)
    merged["best_accuracy_abs_diff"] = (
        merged["best_accuracy_reported"] - merged["best_acc"]
    ).abs()
    merged["total_time_abs_diff_s"] = (
        merged["total_time_reported_s"] - merged["total_time"]
    ).abs()
    return merged


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows, round_rows, errors = [], [], []
    for algorithm in ALGORITHMS:
        for jr_folder in JR_MAP:
            base = args.logs_root / algorithm / jr_folder
            for rep_dir in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name):
                try:
                    run, rounds = parse_run(rep_dir, algorithm, jr_folder, rep_dir.name)
                    run_rows.append(run)
                    round_rows.extend(rounds)
                except Exception as exc:  # retain a complete audit trail
                    errors.append({"path": str(rep_dir), "error": str(exc)})

    runs = pd.DataFrame(run_rows).sort_values(["algorithm", "join_ratio", "rep"])
    rounds = pd.DataFrame(round_rows).sort_values(["algorithm", "join_ratio", "rep", "round"])
    excluded = runs[~runs["planned_run"]].copy()
    planned_runs = runs[runs["planned_run"]].copy()
    planned_rounds = rounds.merge(
        planned_runs[["algorithm", "join_ratio", "rep"]],
        on=["algorithm", "join_ratio", "rep"],
        how="inner",
    )
    validation = compare_with_article(runs, args.article_raw)

    planned_runs.to_csv(args.output_dir / "dissertation_runs.csv", index=False)
    planned_rounds.to_csv(args.output_dir / "dissertation_rounds.csv", index=False)
    excluded.to_csv(args.output_dir / "excluded_additional_runs.csv", index=False)
    validation.to_csv(args.output_dir / "validation_against_article.csv", index=False)
    pd.DataFrame(errors, columns=["path", "error"]).to_csv(args.output_dir / "parsing_errors.csv", index=False)

    checks = {
        "parsed_runs_total": len(runs),
        "planned_runs": len(planned_runs),
        "additional_runs": len(excluded),
        "planned_round_records": len(planned_rounds),
        "parsing_errors": len(errors),
        "all_exit_codes_zero": bool((planned_runs["exit_code"] == 0).all()),
        "learning_rates": sorted(planned_runs["learning_rate"].unique().tolist()),
        "max_best_accuracy_diff_vs_article": float(validation["best_accuracy_abs_diff"].max()),
        "max_total_time_diff_s_vs_article": float(validation["total_time_abs_diff_s"].max()),
        "unmatched_rows_vs_article": int((validation["_merge"] != "both").sum()),
    }
    report = "\n".join(f"{key}: {value}" for key, value in checks.items()) + "\n"
    (args.output_dir / "audit_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
