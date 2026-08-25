#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reutilizamos exatamente o mesmo particionamento federado
# implementado para TON-IoT.
from build_toniot_binary import (
    largest_remainder,
    fixed_size_dirichlet_partition,
    iid_test_partition,
    save_npz,
)


# ---------------------------------------------------------------------
# Feature set PILOTO do CIC-DDoS2019.
#
# Não é necessário ser igual ao TON-IoT. A ideia é preservar
# características próprias do CICFlowMeter, removendo identificadores,
# duplicatas, constantes e atributos já conhecidos por conter Inf.
# ---------------------------------------------------------------------

NUMERIC_FEATURES = [
    "Source Port",
    "Destination Port",
    "Flow Duration",

    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",

    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",

    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",

    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Bwd IAT Mean",
    "Bwd IAT Std",

    "SYN Flag Count",
    "ACK Flag Count",

    "Down/Up Ratio",
    "Average Packet Size",

    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",

    "Active Mean",
    "Active Std",
    "Idle Mean",
    "Idle Std",

    "Inbound",
]

CATEGORICAL_FEATURES = [
    "Protocol",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalize_header_map(fieldnames):
    """
    CIC-DDoS2019 pode conter espaços nos nomes originais das colunas.
    Mapeamos nome normalizado -> nome efetivamente presente no CSV.
    """
    mapping = {}

    for raw_name in fieldnames:
        if raw_name is None:
            continue

        normalized = raw_name.strip()

        if normalized not in mapping:
            mapping[normalized] = raw_name

    return mapping


def reservoir_sample(raw_root, quotas, seed):
    """
    Reservoir sampling independente por Label original.

    Nesta etapa ainda não fazemos conversão numérica das 70M linhas.
    Isso mantém a leitura do piloto viável. As amostras efetivamente
    selecionadas são integralmente validadas antes do preprocessing.
    """
    rng = random.Random(seed)

    reservoirs = {
        label: []
        for label, quota in quotas.items()
        if quota > 0
    }

    seen = Counter()
    total_rows = 0

    files = sorted(raw_root.rglob("*.csv"))

    if not files:
        raise RuntimeError(
            f"Nenhum CSV encontrado em {raw_root}"
        )

    print(f"CSV encontrados: {len(files)}")
    print()

    for file_index, path in enumerate(files, start=1):
        file_rows = 0

        relative = path.relative_to(raw_root)

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
            errors="replace",
        ) as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise RuntimeError(
                    f"CSV sem cabeçalho: {path}"
                )

            header_map = normalize_header_map(
                reader.fieldnames
            )

            required = set(FEATURES + ["Label"])

            missing = sorted(
                required - set(header_map)
            )

            if missing:
                raise RuntimeError(
                    f"{path}: colunas ausentes: {missing}"
                )

            label_key = header_map["Label"]

            feature_keys = {
                feature: header_map[feature]
                for feature in FEATURES
            }

            for row in reader:
                total_rows += 1
                file_rows += 1

                label = (
                    row.get(label_key) or ""
                ).strip()

                if label not in reservoirs:
                    continue

                seen[label] += 1

                quota = quotas[label]
                bucket = reservoirs[label]

                sample = {
                    feature: row.get(raw_key, "")
                    for feature, raw_key
                    in feature_keys.items()
                }

                sample["_attack_label"] = label
                sample["_y"] = (
                    0 if label.upper() == "BENIGN" else 1
                )
                sample["_day"] = path.parent.name
                sample["_file"] = relative.as_posix()

                if len(bucket) < quota:
                    bucket.append(sample)
                else:
                    j = rng.randrange(seen[label])

                    if j < quota:
                        bucket[j] = sample

        print(
            f"[{file_index:02d}/{len(files):02d}] "
            f"{relative.as_posix():<32} "
            f"linhas={file_rows:>10,}"
        )

    print()

    for label, quota in quotas.items():
        obtained = len(reservoirs.get(label, []))

        if obtained != quota:
            raise RuntimeError(
                f"Quota não atingida para {label}: "
                f"esperado={quota}, obtido={obtained}"
            )

    samples = []

    for label in sorted(reservoirs):
        samples.extend(reservoirs[label])

    rng.shuffle(samples)

    return samples, seen, total_rows


def samples_to_arrays(samples):
    """
    Converte e valida integralmente todas as amostras selecionadas.
    """
    numeric = np.empty(
        (len(samples), len(NUMERIC_FEATURES)),
        dtype=np.float64,
    )

    categorical = np.empty(
        (len(samples), len(CATEGORICAL_FEATURES)),
        dtype=object,
    )

    y = np.empty(
        len(samples),
        dtype=np.int64,
    )

    attack_labels = np.empty(
        len(samples),
        dtype=object,
    )

    days = np.empty(
        len(samples),
        dtype=object,
    )

    invalid_rows = 0
    invalid_by_feature = Counter()
    invalid_examples = []

    for i, sample in enumerate(samples):
        row_invalid = False

        for j, feature in enumerate(NUMERIC_FEATURES):
            value = (
                sample.get(feature) or ""
            ).strip()

            try:
                number = float(value)
                valid = math.isfinite(number)
            except (ValueError, TypeError):
                valid = False
                number = np.nan

            if not valid:
                row_invalid = True
                invalid_by_feature[feature] += 1

                if len(invalid_examples) < 20:
                    invalid_examples.append({
                        "feature": feature,
                        "value": value,
                        "label": sample["_attack_label"],
                        "day": sample["_day"],
                        "file": sample["_file"],
                    })

            numeric[i, j] = number

        for j, feature in enumerate(CATEGORICAL_FEATURES):
            value = (
                sample.get(feature) or ""
            ).strip()

            categorical[i, j] = (
                value if value else "__MISSING__"
            )

        y[i] = sample["_y"]
        attack_labels[i] = sample["_attack_label"]
        days[i] = sample["_day"]

        if row_invalid:
            invalid_rows += 1

    if invalid_rows:
        print()
        print("=" * 72)
        print("AMOSTRAS INVÁLIDAS ENCONTRADAS")
        print("=" * 72)
        print(f"Registros inválidos: {invalid_rows:,}")
        print()

        for feature, count in invalid_by_feature.most_common():
            print(
                f"  {feature:<34} {count:>8,}"
            )

        print()
        print("Exemplos:")

        for ex in invalid_examples:
            print(
                f"  {ex['file']} | "
                f"{ex['label']} | "
                f"{ex['feature']}={ex['value']!r}"
            )

        raise RuntimeError(
            "O piloto contém valores inválidos nas features "
            "selecionadas. Investigue antes de gerar o dataset."
        )

    return (
        numeric,
        categorical,
        y,
        attack_labels,
        days,
    )


def preprocess(
    numeric,
    categorical,
    train_idx,
    test_idx,
):
    """
    Fit APENAS no treino para evitar data leakage.
    """
    scaler = StandardScaler()

    x_num_train = scaler.fit_transform(
        numeric[train_idx]
    ).astype(np.float32)

    x_num_test = scaler.transform(
        numeric[test_idx]
    ).astype(np.float32)

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    x_cat_train = encoder.fit_transform(
        categorical[train_idx]
    )

    x_cat_test = encoder.transform(
        categorical[test_idx]
    )

    x_train = np.concatenate(
        [x_num_train, x_cat_train],
        axis=1,
    ).astype(np.float32)

    x_test = np.concatenate(
        [x_num_test, x_cat_test],
        axis=1,
    ).astype(np.float32)

    preprocessing = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "categories": {
            feature: categories.tolist()
            for feature, categories
            in zip(
                CATEGORICAL_FEATURES,
                encoder.categories_,
            )
        },
        "num_output_features": int(
            x_train.shape[1]
        ),
    }

    return (
        x_train,
        x_test,
        preprocessing,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/CIC_DDoS2019/reports/"
            "cicddos2019_inspection.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "code/PFLlib/dataset/"
            "CIC_DDoS2019_pilot"
        ),
    )

    parser.add_argument(
        "--total-samples",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--attack-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--label-concentration",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=20260824,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260825,
    )

    parser.add_argument(
        "--partition-seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.total_samples <= 0:
        raise ValueError(
            "--total-samples deve ser positivo."
        )

    if not (0.0 < args.attack_ratio < 1.0):
        raise ValueError(
            "--attack-ratio deve estar entre 0 e 1."
        )

    if not (0.0 < args.test_size < 1.0):
        raise ValueError(
            "--test-size deve estar entre 0 e 1."
        )

    if args.output.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"{args.output} já existe. "
                "Use --overwrite para regenerar."
            )

        shutil.rmtree(args.output)

    with args.report.open(
        encoding="utf-8"
    ) as f:
        report = json.load(f)

    raw_root = Path(report["root"])

    label_counts = {
        str(key).strip(): int(value)
        for key, value
        in report["label_counts"].items()
    }

    benign_label = None

    for label in label_counts:
        if label.upper() == "BENIGN":
            benign_label = label
            break

    if benign_label is None:
        raise RuntimeError(
            "Label BENIGN não encontrado no relatório."
        )

    normal_target = round(
        args.total_samples
        * (1.0 - args.attack_ratio)
    )

    attack_target = (
        args.total_samples - normal_target
    )

    attack_counts = {
        label: count
        for label, count in label_counts.items()
        if label != benign_label
    }

    attack_quotas = largest_remainder(
        attack_target,
        attack_counts,
    )

    quotas = {
        benign_label: normal_target,
        **attack_quotas,
    }

    print("=" * 72)
    print("CIC-DDoS2019 BINARY PILOT")
    print("=" * 72)
    print(
        f"Dataset original: "
        f"{report['total_rows']:,}"
    )
    print(
        f"Amostras piloto: "
        f"{args.total_samples:,}"
    )
    print(
        f"Normal: {normal_target:,} "
        f"({normal_target / args.total_samples:.2%})"
    )
    print(
        f"Ataque: {attack_target:,} "
        f"({attack_target / args.total_samples:.2%})"
    )
    print()
    print("Quotas por Label original:")

    for label, value in sorted(
        quotas.items(),
        key=lambda item: (
            item[0].upper() != "BENIGN",
            -item[1],
            item[0],
        ),
    ):
        print(
            f"  {label:<22} {value:>7,}"
        )

    print()
    zero_quota_attacks = [
        label
        for label, value in attack_quotas.items()
        if value == 0
    ]

    if zero_quota_attacks:
        print(
            "Aviso: subtipos com quota zero neste piloto "
            "devido ao tamanho reduzido:"
        )

        for label in zero_quota_attacks:
            print(f"  {label}")

        print()

    print("Iniciando reservoir sampling...")
    print()

    (
        samples,
        seen,
        scanned_rows,
    ) = reservoir_sample(
        raw_root,
        quotas,
        args.sampling_seed,
    )

    print(
        f"Linhas examinadas: {scanned_rows:,}"
    )

    (
        numeric,
        categorical,
        y,
        attack_labels,
        days,
    ) = samples_to_arrays(samples)

    print(
        "Validação das features selecionadas: OK"
    )

    all_idx = np.arange(len(y))

    # PILOTO:
    # split aleatório estratificado apenas pela classe binária.
    #
    # Antes da campanha definitiva será avaliado se o split temporal
    # original 01-12 / 03-11 deve ser preservado.
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=args.test_size,
        random_state=args.split_seed,
        shuffle=True,
        stratify=y,
    )

    print()
    print(
        f"Train global: {len(train_idx):,}"
    )
    print(
        f"Test global:  {len(test_idx):,}"
    )

    (
        x_train,
        x_test,
        preprocessing,
    ) = preprocess(
        numeric,
        categorical,
        train_idx,
        test_idx,
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    attack_labels_train = attack_labels[
        train_idx
    ]
    attack_labels_test = attack_labels[
        test_idx
    ]

    days_train = days[train_idx]
    days_test = days[test_idx]

    train_clients = fixed_size_dirichlet_partition(
        y_train,
        args.num_clients,
        args.label_concentration,
        args.partition_seed,
    )

    test_clients = iid_test_partition(
        y_test,
        args.num_clients,
        args.split_seed,
    )

    train_dir = args.output / "train"
    test_dir = args.output / "test"

    train_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics = []

    print()
    print("Clientes:")
    print("-" * 72)

    for client_id in range(args.num_clients):
        tr_idx = train_clients[client_id]
        te_idx = test_clients[client_id]

        save_npz(
            train_dir / f"{client_id}.npz",
            x_train[tr_idx],
            y_train[tr_idx],
        )

        save_npz(
            test_dir / f"{client_id}.npz",
            x_test[te_idx],
            y_test[te_idx],
        )

        client_stats = [
            [
                int(label),
                int(
                    (
                        y_train[tr_idx] == label
                    ).sum()
                ),
            ]
            for label
            in np.unique(y_train[tr_idx])
        ]

        statistics.append(client_stats)

        print(
            f"Cliente {client_id:02d}: "
            f"train={len(tr_idx):>5,} "
            f"test={len(te_idx):>4,} "
            f"labels={client_stats}"
        )

    config = {
        "dataset": "CIC_DDoS2019",
        "variant": "binary_pilot",
        "num_clients": args.num_clients,
        "num_classes": 2,
        "num_features": int(
            x_train.shape[1]
        ),
        "non_iid": True,
        "balance": True,
        "partition":
            "fixed_size_dirichlet_label_skew",
        "alpha": None,
        "label_concentration":
            args.label_concentration,
        "batch_size": 10,

        "total_samples":
            args.total_samples,
        "attack_ratio":
            args.attack_ratio,
        "test_size":
            args.test_size,

        "sampling_seed":
            args.sampling_seed,
        "split_seed":
            args.split_seed,
        "partition_seed":
            args.partition_seed,

        "split_mode":
            "random_stratified_binary_pilot",

        "raw_numeric_features":
            NUMERIC_FEATURES,
        "raw_categorical_features":
            CATEGORICAL_FEATURES,

        "attack_sampling_quotas":
            quotas,

        "sample_attack_label_counts": {
            str(key): int(value)
            for key, value
            in Counter(attack_labels).items()
        },

        "train_attack_label_counts": {
            str(key): int(value)
            for key, value
            in Counter(
                attack_labels_train
            ).items()
        },

        "test_attack_label_counts": {
            str(key): int(value)
            for key, value
            in Counter(
                attack_labels_test
            ).items()
        },

        "sample_day_counts": {
            str(key): int(value)
            for key, value
            in Counter(days).items()
        },

        "train_day_counts": {
            str(key): int(value)
            for key, value
            in Counter(days_train).items()
        },

        "test_day_counts": {
            str(key): int(value)
            for key, value
            in Counter(days_test).items()
        },

        "Size of samples for labels in clients":
            statistics,
    }

    with (
        args.output / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (
        args.output / "preprocessing.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            preprocessing,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("PILOTO GERADO COM SUCESSO")
    print("=" * 72)
    print(
        f"Saída:        {args.output}"
    )
    print(
        f"Input DNN:    "
        f"{x_train.shape[1]} features"
    )
    print(
        f"Train shape:  {x_train.shape}"
    )
    print(
        f"Test shape:   {x_test.shape}"
    )

    print()
    print("Origem temporal das amostras:")
    print(
        "  Total: ",
        dict(Counter(days))
    )
    print(
        "  Train: ",
        dict(Counter(days_train))
    )
    print(
        "  Test:  ",
        dict(Counter(days_test))
    )


if __name__ == "__main__":
    main()
