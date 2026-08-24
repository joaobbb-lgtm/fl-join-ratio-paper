#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "src_port",
    "dst_port",
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "src_ip_bytes",
    "dst_ip_bytes",
    "missed_bytes",
]

CATEGORICAL_FEATURES = [
    "proto",
    "conn_state",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def largest_remainder(total, counts):
    """Distribui exatamente 'total' proporcionalmente a counts."""
    count_sum = sum(counts.values())

    raw = {
        key: total * value / count_sum
        for key, value in counts.items()
    }

    quotas = {
        key: int(math.floor(value))
        for key, value in raw.items()
    }

    remaining = total - sum(quotas.values())

    order = sorted(
        counts,
        key=lambda key: raw[key] - quotas[key],
        reverse=True,
    )

    for key in order[:remaining]:
        quotas[key] += 1

    return quotas


def reservoir_sample(raw_root, quotas, seed):
    """
    Amostragem reservoir independente por subtipo.
    Garante exatamente a quota desejada de cada tipo.
    """
    rng = random.Random(seed)

    reservoirs = {
        attack_type: []
        for attack_type, quota in quotas.items()
        if quota > 0
    }

    seen = Counter()
    label_mismatches = 0
    total_rows = 0

    invalid_rows = 0
    invalid_by_feature = Counter()
    invalid_by_type = Counter()
    invalid_by_file = Counter()

    files = sorted(raw_root.glob("*.csv"))

    if not files:
        raise RuntimeError(
            f"Nenhum CSV encontrado em {raw_root}"
        )

    print(f"CSV encontrados: {len(files)}")
    print()

    for file_index, path in enumerate(files, start=1):
        file_rows = 0

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
            errors="replace",
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                total_rows += 1
                file_rows += 1

                attack_type = row["type"].strip().lower()

                if attack_type not in reservoirs:
                    continue

                expected_label = (
                    0 if attack_type == "normal" else 1
                )

                try:
                    original_label = int(float(row["label"]))
                except (ValueError, TypeError):
                    original_label = expected_label

                if original_label != expected_label:
                    label_mismatches += 1

                invalid_features = []

                for feature in NUMERIC_FEATURES:
                    value = (row.get(feature) or "").strip()

                    try:
                        number = float(value)
                        valid = math.isfinite(number)
                    except (ValueError, TypeError):
                        valid = False

                    if not valid:
                        invalid_features.append(feature)

                if invalid_features:
                    invalid_rows += 1
                    invalid_by_type[attack_type] += 1
                    invalid_by_file[path.name] += 1

                    for feature in invalid_features:
                        invalid_by_feature[feature] += 1

                    continue

                # Incrementar somente depois da validação:
                # o reservoir deve operar sobre a população válida.
                seen[attack_type] += 1

                sample = {
                    feature: row.get(feature, "")
                    for feature in FEATURES
                }

                sample["_type"] = attack_type
                sample["_y"] = expected_label

                quota = quotas[attack_type]
                bucket = reservoirs[attack_type]

                if len(bucket) < quota:
                    bucket.append(sample)
                else:
                    j = rng.randrange(seen[attack_type])

                    if j < quota:
                        bucket[j] = sample

        print(
            f"[{file_index:02d}/{len(files):02d}] "
            f"{path.name:<24} "
            f"linhas={file_rows:>10,}"
        )

    print()

    for attack_type, quota in quotas.items():
        obtained = len(reservoirs.get(attack_type, []))

        if obtained != quota:
            raise RuntimeError(
                f"Quota não atingida para {attack_type}: "
                f"esperado={quota}, obtido={obtained}"
            )

    samples = []

    for attack_type in sorted(reservoirs):
        samples.extend(reservoirs[attack_type])

    rng.shuffle(samples)

    invalid_summary = {
        "rows": invalid_rows,
        "by_feature": dict(invalid_by_feature),
        "by_type": dict(invalid_by_type),
        "by_file": dict(invalid_by_file),
    }

    return (
        samples,
        seen,
        label_mismatches,
        total_rows,
        invalid_summary,
    )


def samples_to_arrays(samples):
    numeric = np.empty(
        (len(samples), len(NUMERIC_FEATURES)),
        dtype=np.float64,
    )

    categorical = np.empty(
        (len(samples), len(CATEGORICAL_FEATURES)),
        dtype=object,
    )

    y = np.empty(len(samples), dtype=np.int64)
    types = np.empty(len(samples), dtype=object)

    for i, sample in enumerate(samples):
        for j, feature in enumerate(NUMERIC_FEATURES):
            value = sample[feature].strip()

            if value == "":
                numeric[i, j] = np.nan
            else:
                numeric[i, j] = float(value)

        for j, feature in enumerate(CATEGORICAL_FEATURES):
            value = sample[feature].strip()

            categorical[i, j] = (
                value if value else "__MISSING__"
            )

        y[i] = sample["_y"]
        types[i] = sample["_type"]

    if not np.isfinite(numeric).all():
        bad = int((~np.isfinite(numeric)).sum())

        raise RuntimeError(
            f"Foram encontrados {bad} valores numéricos "
            "não finitos nas features selecionadas."
        )

    return numeric, categorical, y, types


def preprocess(
    numeric,
    categorical,
    train_idx,
    test_idx,
):
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
        "num_output_features": int(x_train.shape[1]),
    }

    return x_train, x_test, preprocessing


def fixed_size_dirichlet_partition(
    y,
    num_clients,
    concentration,
    seed,
):
    """
    Label-skew Dirichlet/Beta com tamanho controlado por cliente.

    Em classificação binária, a proporção local de ataques é
    amostrada de uma Beta centrada na proporção global das classes.

    O número de amostras por cliente permanece igual (ou difere
    em no máximo uma amostra quando a divisão não é exata).
    """
    rng = np.random.default_rng(seed)

    y = np.asarray(y)

    labels = np.unique(y)

    if not np.array_equal(labels, np.array([0, 1])):
        raise ValueError(
            "Este particionador requer classificação binária "
            "com labels 0 e 1."
        )

    n = len(y)

    client_sizes = np.full(
        num_clients,
        n // num_clients,
        dtype=np.int64,
    )

    client_sizes[: n % num_clients] += 1

    attack_total = int((y == 1).sum())
    normal_total = int((y == 0).sum())

    attack_ratio = attack_total / n

    beta_a = attack_ratio * concentration
    beta_b = (1.0 - attack_ratio) * concentration

    if beta_a <= 0 or beta_b <= 0:
        raise ValueError(
            "A distribuição global deve conter as duas classes."
        )

    # Procuramos uma amostragem sem clientes de classe única.
    for attempt in range(1, 1001):
        probabilities = rng.beta(
            beta_a,
            beta_b,
            size=num_clients,
        )

        # Ajuste por deslocamento no espaço logit para preservar
        # exatamente a proporção global de ataques.
        eps = 1e-12

        probabilities = np.clip(
            probabilities,
            eps,
            1.0 - eps,
        )

        logits = np.log(
            probabilities / (1.0 - probabilities)
        )

        lo = -50.0
        hi = 50.0

        for _ in range(100):
            mid = (lo + hi) / 2.0

            adjusted = 1.0 / (
                1.0 + np.exp(-(logits + mid))
            )

            expected_attacks = float(
                np.sum(client_sizes * adjusted)
            )

            if expected_attacks < attack_total:
                lo = mid
            else:
                hi = mid

        probabilities = 1.0 / (
            1.0
            + np.exp(
                -(logits + (lo + hi) / 2.0)
            )
        )

        raw_attack_counts = (
            probabilities * client_sizes
        )

        attack_counts = np.floor(
            raw_attack_counts
        ).astype(np.int64)

        remaining = (
            attack_total - int(attack_counts.sum())
        )

        if remaining > 0:
            fractions = (
                raw_attack_counts - attack_counts
            )

            order = np.argsort(-fractions)

            for client_id in order:
                if remaining == 0:
                    break

                if (
                    attack_counts[client_id]
                    < client_sizes[client_id]
                ):
                    attack_counts[client_id] += 1
                    remaining -= 1

        normal_counts = (
            client_sizes - attack_counts
        )

        # Evitar clientes exclusivamente normal ou ataque.
        if (
            np.all(attack_counts > 0)
            and np.all(normal_counts > 0)
        ):
            break
    else:
        raise RuntimeError(
            "Não foi possível gerar uma partição sem "
            "clientes de classe única."
        )

    if int(attack_counts.sum()) != attack_total:
        raise RuntimeError(
            "Número global de ataques não foi preservado."
        )

    if int(normal_counts.sum()) != normal_total:
        raise RuntimeError(
            "Número global de normais não foi preservado."
        )

    attack_indices = np.where(y == 1)[0]
    normal_indices = np.where(y == 0)[0]

    rng.shuffle(attack_indices)
    rng.shuffle(normal_indices)

    clients = []

    attack_cursor = 0
    normal_cursor = 0

    for client_id in range(num_clients):
        n_attack = int(attack_counts[client_id])
        n_normal = int(normal_counts[client_id])

        client_attack = attack_indices[
            attack_cursor:
            attack_cursor + n_attack
        ]

        client_normal = normal_indices[
            normal_cursor:
            normal_cursor + n_normal
        ]

        attack_cursor += n_attack
        normal_cursor += n_normal

        indices = np.concatenate(
            [client_normal, client_attack]
        )

        rng.shuffle(indices)

        clients.append(
            indices.astype(np.int64)
        )

    attack_pct = (
        attack_counts / client_sizes * 100.0
    )

    percentiles = np.percentile(
        attack_pct,
        [0, 10, 25, 50, 75, 90, 100],
    )

    print(
        "Partição label-skew de tamanho controlado "
        f"concluída após {attempt} tentativa(s)."
    )
    print(
        f"Concentração kappa: {concentration}"
    )
    print(
        f"Tamanho dos clientes: "
        f"{client_sizes.min()} a {client_sizes.max()}"
    )
    print(
        "Ataques por cliente (%): "
        f"min={percentiles[0]:.2f}, "
        f"p10={percentiles[1]:.2f}, "
        f"p25={percentiles[2]:.2f}, "
        f"mediana={percentiles[3]:.2f}, "
        f"p75={percentiles[4]:.2f}, "
        f"p90={percentiles[5]:.2f}, "
        f"max={percentiles[6]:.2f}"
    )

    return clients


def iid_test_partition(
    y,
    num_clients,
    seed,
):
    """
    Divide o teste de forma fixa e aproximadamente IID,
    preservando cada classe entre os clientes.
    """
    rng = np.random.default_rng(seed)

    client_indices = [
        []
        for _ in range(num_clients)
    ]

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)

        chunks = np.array_split(idx, num_clients)

        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(
                chunk.tolist()
            )

    for indices in client_indices:
        rng.shuffle(indices)

    return [
        np.asarray(indices, dtype=np.int64)
        for indices in client_indices
    ]


def save_npz(path, x, y):
    data = {
        "x": x.astype(np.float32),
        "y": y.astype(np.int64),
    }

    np.savez_compressed(path, data=data)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/TON_IoT/reports/"
            "toniot_network_inspection.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "code/PFLlib/dataset/TON_IoT_pilot"
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
        default=10,
    )

    parser.add_argument(
        "--label-concentration",
        type=float,
        default=5.0,
        help=(
            "Concentração kappa da distribuição Beta/Dirichlet "
            "usada para gerar label skew com tamanho controlado."
        ),
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

    args = parser.parse_args()

    with args.report.open(
        encoding="utf-8"
    ) as f:
        report = json.load(f)

    raw_root = Path(report["root"])

    type_counts = {
        key.lower(): int(value)
        for key, value
        in report["type_counts"].items()
    }

    normal_target = round(
        args.total_samples
        * (1.0 - args.attack_ratio)
    )

    attack_target = (
        args.total_samples - normal_target
    )

    attack_counts = {
        key: value
        for key, value in type_counts.items()
        if key != "normal"
    }

    attack_quotas = largest_remainder(
        attack_target,
        attack_counts,
    )

    quotas = {
        "normal": normal_target,
        **attack_quotas,
    }

    print("=" * 72)
    print("TON-IoT BINARY PILOT")
    print("=" * 72)
    print(f"Dataset original: {report['total_rows']:,}")
    print(f"Amostras piloto: {args.total_samples:,}")
    print(
        f"Normal: {normal_target:,} "
        f"({normal_target / args.total_samples:.2%})"
    )
    print(
        f"Ataque: {attack_target:,} "
        f"({attack_target / args.total_samples:.2%})"
    )
    print()
    print("Quotas por tipo:")

    for key, value in quotas.items():
        print(f"  {key:<12} {value:>7,}")

    print()
    print("Iniciando reservoir sampling...")
    print()

    (
        samples,
        seen,
        label_mismatches,
        scanned_rows,
        invalid_summary,
    ) = reservoir_sample(
        raw_root,
        quotas,
        args.sampling_seed,
    )

    print(
        f"Linhas examinadas: {scanned_rows:,}"
    )
    print(
        f"Inconsistências type/label: "
        f"{label_mismatches:,}"
    )
    print(
        f"Registros inválidos descartados: "
        f"{invalid_summary['rows']:,}"
    )

    if invalid_summary["rows"]:
        print("Inválidos por feature:")
        for feature, count in sorted(
            invalid_summary["by_feature"].items()
        ):
            print(f"  {feature:<16} {count:>8,}")

        print("Inválidos por type:")
        for attack_type, count in sorted(
            invalid_summary["by_type"].items()
        ):
            print(f"  {attack_type:<16} {count:>8,}")

    (
        numeric,
        categorical,
        y,
        types,
    ) = samples_to_arrays(samples)

    all_idx = np.arange(len(y))

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

    x_train, x_test, preprocessing = preprocess(
        numeric,
        categorical,
        train_idx,
        test_idx,
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    type_train = types[train_idx]
    type_test = types[test_idx]

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
                int((y_train[tr_idx] == label).sum()),
            ]
            for label in np.unique(y_train[tr_idx])
        ]

        statistics.append(client_stats)

        print(
            f"Cliente {client_id:02d}: "
            f"train={len(tr_idx):>5,} "
            f"test={len(te_idx):>4,} "
            f"labels={client_stats}"
        )

    config = {
        "dataset": "TON_IoT",
        "variant": "binary_pilot",
        "num_clients": args.num_clients,
        "num_classes": 2,
        "num_features": int(x_train.shape[1]),
        "non_iid": True,
        "balance": True,
        "partition": "fixed_size_dirichlet_label_skew",
        "alpha": None,
        "label_concentration": args.label_concentration,
        "batch_size": 10,
        "total_samples": args.total_samples,
        "attack_ratio": args.attack_ratio,
        "test_size": args.test_size,
        "sampling_seed": args.sampling_seed,
        "split_seed": args.split_seed,
        "partition_seed": args.partition_seed,
        "raw_numeric_features": NUMERIC_FEATURES,
        "raw_categorical_features": CATEGORICAL_FEATURES,
        "attack_sampling_quotas": quotas,
        "invalid_rows_discarded": invalid_summary["rows"],
        "invalid_rows_by_feature": invalid_summary["by_feature"],
        "invalid_rows_by_type": invalid_summary["by_type"],
        "invalid_rows_by_file": invalid_summary["by_file"],
        "train_type_counts": {
            key: int(value)
            for key, value
            in Counter(type_train).items()
        },
        "test_type_counts": {
            key: int(value)
            for key, value
            in Counter(type_test).items()
        },
        "Size of samples for labels in clients": statistics,
    }

    with (
        args.output / "config.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (
        args.output / "preprocessing.json"
    ).open("w", encoding="utf-8") as f:
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
    print(f"Saída:        {args.output}")
    print(
        f"Input DNN:    {x_train.shape[1]} features"
    )
    print(
        f"Train shape:  {x_train.shape}"
    )
    print(
        f"Test shape:   {x_test.shape}"
    )


if __name__ == "__main__":
    main()
