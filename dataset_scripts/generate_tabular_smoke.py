#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np


SEED = 20260824
NUM_CLIENTS = 10
NUM_FEATURES = 60
NUM_CLASSES = 2
TRAIN_PER_CLIENT = 100
TEST_PER_CLIENT = 40

OUTPUT = Path("code/PFLlib/dataset/TabularSmoke")


def make_samples(rng, n_samples):
    y = rng.integers(0, NUM_CLASSES, size=n_samples, dtype=np.int64)

    x = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, NUM_FEATURES),
    ).astype(np.float32)

    # Torna o problema propositalmente aprendível:
    # classe 1 desloca as primeiras features.
    x[y == 1, :8] += 1.5

    return x, y


def save_npz(path, x, y):
    data = {
        "x": x.astype(np.float32),
        "y": y.astype(np.int64),
    }

    np.savez_compressed(path, data=data)


def main():
    rng = np.random.default_rng(SEED)

    train_dir = OUTPUT / "train"
    test_dir = OUTPUT / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    statistics = []

    for client_id in range(NUM_CLIENTS):
        x_train, y_train = make_samples(rng, TRAIN_PER_CLIENT)
        x_test, y_test = make_samples(rng, TEST_PER_CLIENT)

        save_npz(
            train_dir / f"{client_id}.npz",
            x_train,
            y_train,
        )

        save_npz(
            test_dir / f"{client_id}.npz",
            x_test,
            y_test,
        )

        client_stats = [
            [int(label), int((y_train == label).sum())]
            for label in range(NUM_CLASSES)
        ]
        statistics.append(client_stats)

        print(
            f"Cliente {client_id:2d}: "
            f"train={len(y_train):3d} "
            f"test={len(y_test):2d} "
            f"labels_train={client_stats}"
        )

    config = {
        "num_clients": NUM_CLIENTS,
        "num_classes": NUM_CLASSES,
        "non_iid": False,
        "balance": True,
        "partition": "smoke",
        "Size of samples for labels in clients": statistics,
        "alpha": None,
        "batch_size": 10,
        "num_features": NUM_FEATURES,
        "seed": SEED,
        "purpose": "Tabular PFLlib integration smoke test",
    }

    with (OUTPUT / "config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Dataset criado em: {OUTPUT}")
    print(
        f"Shape esperado por amostra: ({NUM_FEATURES},)"
    )


if __name__ == "__main__":
    main()
