#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


MISSING_TOKENS = {"", "-", "(empty)", "null", "none"}


def natural_file_number(path):
    return int(path.stem.split("_")[-1])


class ColumnProfile:
    def __init__(self, unique_cap=10000):
        self.sampled = 0
        self.missing = 0
        self.nan = 0
        self.pos_inf = 0
        self.neg_inf = 0

        self.numeric = 0
        self.non_numeric = 0

        self.numeric_min = None
        self.numeric_max = None

        self.unique_cap = unique_cap
        self.unique_values = set()
        self.unique_truncated = False

        self.examples = []

    def add(self, raw_value):
        self.sampled += 1

        value = "" if raw_value is None else raw_value.strip()
        lower = value.lower()

        if lower in MISSING_TOKENS:
            self.missing += 1
            return

        if lower == "nan":
            self.nan += 1
            self.numeric += 1
            return

        try:
            numeric_value = float(value)
            self.numeric += 1

            if math.isinf(numeric_value):
                if numeric_value > 0:
                    self.pos_inf += 1
                else:
                    self.neg_inf += 1
            elif not math.isnan(numeric_value):
                if self.numeric_min is None or numeric_value < self.numeric_min:
                    self.numeric_min = numeric_value

                if self.numeric_max is None or numeric_value > self.numeric_max:
                    self.numeric_max = numeric_value

        except ValueError:
            self.non_numeric += 1

        if not self.unique_truncated:
            self.unique_values.add(value)

            if len(self.unique_values) > self.unique_cap:
                self.unique_truncated = True
                self.unique_values.clear()

        if len(self.examples) < 5 and value not in self.examples:
            self.examples.append(value)

    def as_dict(self):
        observed = self.numeric + self.non_numeric

        if observed == 0:
            inferred_type = "empty"
            numeric_ratio = None
        else:
            numeric_ratio = self.numeric / observed

            if numeric_ratio >= 0.98:
                inferred_type = "numeric"
            elif numeric_ratio <= 0.02:
                inferred_type = "categorical"
            else:
                inferred_type = "mixed"

        if self.unique_truncated:
            unique_description = f">{self.unique_cap}"
        else:
            unique_description = len(self.unique_values)

        return {
            "sampled": self.sampled,
            "missing": self.missing,
            "nan": self.nan,
            "positive_inf": self.pos_inf,
            "negative_inf": self.neg_inf,
            "numeric_values": self.numeric,
            "non_numeric_values": self.non_numeric,
            "numeric_ratio": numeric_ratio,
            "inferred_type": inferred_type,
            "numeric_min": self.numeric_min,
            "numeric_max": self.numeric_max,
            "unique_sampled": unique_description,
            "examples": self.examples,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Inspeciona os 23 CSVs do TON-IoT Processed Network Dataset."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/TON_IoT/raw/Processed_Network_dataset"),
        help="Diretório contendo Network_dataset_*.csv",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=40,
        help=(
            "Perfila uma linha a cada N registros. "
            "Os labels continuam sendo contados em todas as linhas."
        ),
    )

    parser.add_argument(
        "--unique-cap",
        type=int,
        default=10000,
        help="Limite para armazenar valores únicos por feature.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/TON_IoT/reports/toniot_network_inspection.txt"),
        help="Arquivo de relatório textual.",
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=Path("data/TON_IoT/reports/toniot_network_inspection.json"),
        help="Arquivo JSON com os resultados.",
    )

    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride deve ser >= 1")

    files = sorted(
        args.root.glob("Network_dataset_*.csv"),
        key=natural_file_number,
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhum Network_dataset_*.csv encontrado em {args.root}"
        )

    total_rows = 0
    malformed_rows = 0
    label_counts = Counter()
    type_counts = Counter()

    file_summaries = []

    feature_names = None
    profiles = {}

    print(f"Arquivos encontrados: {len(files)}")
    print(f"Amostragem de features: 1 a cada {args.stride} registros")
    print()

    for file_index, path in enumerate(files, start=1):
        file_rows = 0
        file_bad = 0
        file_labels = Counter()
        file_types = Counter()

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
            errors="replace",
        ) as f:
            reader = csv.DictReader(f)

            header = reader.fieldnames or []

            normalized_header = [
                column for column in header if column != "uid"
            ]

            if feature_names is None:
                feature_names = [
                    column
                    for column in normalized_header
                    if column not in {"label", "type"}
                ]

                profiles = {
                    column: ColumnProfile(args.unique_cap)
                    for column in feature_names
                }

            for row_index, row in enumerate(reader, start=1):
                if row.get(None):
                    file_bad += 1
                    continue

                label = (row.get("label") or "").strip()
                attack_type = (row.get("type") or "").strip()

                file_labels[label] += 1
                file_types[attack_type] += 1
                file_rows += 1

                if (row_index - 1) % args.stride == 0:
                    for column in feature_names:
                        profiles[column].add(row.get(column))

        total_rows += file_rows
        malformed_rows += file_bad
        label_counts.update(file_labels)
        type_counts.update(file_types)

        file_summaries.append(
            {
                "file": path.name,
                "columns": len(header),
                "has_uid": "uid" in header,
                "rows": file_rows,
                "malformed_rows": file_bad,
                "labels": dict(file_labels),
                "types": dict(file_types),
            }
        )

        print(
            f"[{file_index:02d}/{len(files):02d}] "
            f"{path.name:24s} "
            f"registros={file_rows:9,d} "
            f"colunas={len(header):2d} "
            f"uid={'SIM' if 'uid' in header else 'não'}"
        )

    feature_results = {
        column: profile.as_dict()
        for column, profile in profiles.items()
    }

    result = {
        "dataset": "TON_IoT Processed Network Dataset",
        "root": str(args.root),
        "files": len(files),
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "feature_sampling_stride": args.stride,
        "feature_count_after_uid_removal": len(feature_names),
        "label_counts": dict(label_counts),
        "type_counts": dict(type_counts),
        "file_summaries": file_summaries,
        "features": feature_results,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)

    with args.json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = []

    lines.append("=" * 78)
    lines.append("TON-IoT NETWORK DATASET - RELATÓRIO DE INSPEÇÃO")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Diretório:                  {args.root}")
    lines.append(f"Arquivos:                   {len(files)}")
    lines.append(f"Registros válidos:          {total_rows:,}")
    lines.append(f"Linhas problemáticas:       {malformed_rows:,}")
    lines.append(
        f"Features após remover uid:  {len(feature_names)}"
    )
    lines.append(
        f"Amostragem de features:     1 a cada {args.stride} registros"
    )

    lines.append("")
    lines.append("LABEL")
    lines.append("-" * 78)

    for value, count in label_counts.most_common():
        pct = 100 * count / total_rows
        lines.append(
            f"{value!r:20s} {count:12,d} ({pct:8.4f}%)"
        )

    lines.append("")
    lines.append("TYPE")
    lines.append("-" * 78)

    for value, count in type_counts.most_common():
        pct = 100 * count / total_rows
        lines.append(
            f"{value!r:20s} {count:12,d} ({pct:8.4f}%)"
        )

    lines.append("")
    lines.append("FEATURES - PERFIL DA AMOSTRA")
    lines.append("-" * 78)

    for column in feature_names:
        profile = feature_results[column]

        sampled = profile["sampled"]
        missing_pct = (
            100 * profile["missing"] / sampled
            if sampled
            else 0.0
        )

        numeric_ratio = profile["numeric_ratio"]

        if numeric_ratio is None:
            numeric_text = "n/a"
        else:
            numeric_text = f"{100 * numeric_ratio:.2f}%"

        lines.append("")
        lines.append(f"[{column}]")
        lines.append(
            f"  tipo inferido:        {profile['inferred_type']}"
        )
        lines.append(
            f"  amostras analisadas:  {sampled:,}"
        )
        lines.append(
            f"  ausentes:             {profile['missing']:,} "
            f"({missing_pct:.2f}%)"
        )
        lines.append(
            f"  NaN:                  {profile['nan']:,}"
        )
        lines.append(
            f"  +Inf / -Inf:          "
            f"{profile['positive_inf']:,} / "
            f"{profile['negative_inf']:,}"
        )
        lines.append(
            f"  proporção numérica:   {numeric_text}"
        )
        lines.append(
            f"  valores únicos*:      {profile['unique_sampled']}"
        )

        if profile["numeric_min"] is not None:
            lines.append(
                f"  mínimo numérico:      {profile['numeric_min']}"
            )
            lines.append(
                f"  máximo numérico:      {profile['numeric_max']}"
            )

        lines.append(
            f"  exemplos:             {profile['examples']}"
        )

    lines.append("")
    lines.append("* Cardinalidade calculada sobre a amostra de features.")
    lines.append(
        f"  Valores >{args.unique_cap} indicam alta cardinalidade."
    )
    lines.append("")

    report_text = "\n".join(lines)

    with args.report.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print()
    print("=" * 78)
    print(f"Registros válidos: {total_rows:,}")
    print(f"Linhas problemáticas: {malformed_rows:,}")
    print()
    print(f"Relatório: {args.report}")
    print(f"JSON:      {args.json}")


if __name__ == "__main__":
    main()
