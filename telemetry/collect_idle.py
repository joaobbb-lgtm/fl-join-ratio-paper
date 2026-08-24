#!/usr/bin/env python3

import argparse
import asyncio
import csv
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from kasa import Discover, Credentials


DEFAULT_TAPO_IP = "192.168.0.127"
DEFAULT_INTERVAL = 1.0
DEFAULT_DURATION = 180.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coleta sincronizada de telemetria do sistema e da Tapo P110."
    )
    parser.add_argument("--output", type=Path, help="Arquivo CSV de saída.")
    parser.add_argument("--ready-file", type=Path, help="Marcador criado após a primeira amostra.")
    parser.add_argument("--experiment-id", default="idle")
    parser.add_argument("--tapo-ip", default=os.environ.get("TAPO_IP", DEFAULT_TAPO_IP))
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Duração em segundos; use 0 para coletar até receber SIGINT/SIGTERM.",
    )
    return parser.parse_args()


def gpu_stats():
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,power.draw,memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]

    output = subprocess.check_output(command, text=True).strip()
    values = [v.strip() for v in output.split(",")]

    return {
        "gpu_util_percent": float(values[0]),
        "gpu_power_w": float(values[1]),
        "gpu_memory_mib": float(values[2]),
        "gpu_temp_c": float(values[3]),
    }


async def main():
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval deve ser maior que zero.")
    if args.duration < 0:
        raise ValueError("--duration não pode ser negativa.")

    username = os.environ["KASA_USERNAME"]
    password = os.environ["KASA_PASSWORD"]

    credentials = Credentials(username, password)

    device = await Discover.discover_single(
        args.tapo_ip,
        credentials=credentials,
    )

    if device is None:
        raise RuntimeError("Tapo P110 não encontrada.")

    output_file = args.output or Path(
        "telemetry/logs/"
        + args.experiment_id
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.ready_file:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.unlink(missing_ok=True)

    fields = [
        "timestamp_local",
        "timestamp_utc",
        "epoch_ms",
        "elapsed_s",
        "experiment_id",
        "tapo_power_w",
        "voltage_v",
        "current_a",
        "cpu_percent",
        "ram_percent",
        "gpu_util_percent",
        "gpu_power_w",
        "gpu_memory_mib",
        "gpu_temp_c",
    ]

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    start = time.monotonic()

    try:
        with output_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            while not stop_requested:
                elapsed = time.monotonic() - start

                if args.duration and elapsed >= args.duration:
                    break

                await device.update()

                now = datetime.now().astimezone()
                now_utc = now.astimezone(timezone.utc)
                epoch_ms = time.time_ns() // 1_000_000
                tapo_power = device.features["current_consumption"].value
                voltage = device.features["voltage"].value
                current = device.features["current"].value

                gpu = gpu_stats()
                row = {
                    "timestamp_local": now.isoformat(timespec="milliseconds"),
                    "timestamp_utc": now_utc.isoformat(timespec="milliseconds"),
                    "epoch_ms": epoch_ms,
                    "elapsed_s": round(elapsed, 3),
                    "experiment_id": args.experiment_id,
                    "tapo_power_w": tapo_power,
                    "voltage_v": voltage,
                    "current_a": current,
                    "cpu_percent": psutil.cpu_percent(),
                    "ram_percent": psutil.virtual_memory().percent,
                    **gpu,
                }

                writer.writerow(row)
                f.flush()

                if args.ready_file and not args.ready_file.exists():
                    args.ready_file.touch()

                print(
                    f"{now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
                    f"{elapsed:6.1f}s | "
                    f"Tapo {tapo_power:6.1f} W | "
                    f"CPU {row['cpu_percent']:5.1f}% | "
                    f"GPU {gpu['gpu_util_percent']:5.1f}% | "
                    f"GPU Power {gpu['gpu_power_w']:5.1f} W"
                )

                await asyncio.sleep(args.interval)
    finally:
        await device.disconnect()

    print()
    print(f"Coleta concluída: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
