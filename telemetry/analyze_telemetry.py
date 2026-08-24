#!/usr/bin/env python3

import argparse
import csv
import contextlib
import statistics
from pathlib import Path


def mean(values):
    return statistics.fmean(values)


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def integrate_trapezoid(timestamps_ms, powers_w):
    energy_j = 0.0

    for i in range(1, len(timestamps_ms)):
        dt_s = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0
        avg_power_w = (powers_w[i - 1] + powers_w[i]) / 2.0
        energy_j += avg_power_w * dt_s

    return energy_j


def parse_args():
    parser = argparse.ArgumentParser(description="Resume uma coleta de telemetria.")
    parser.add_argument("csv", type=Path, help="Arquivo CSV coletado.")
    parser.add_argument("--output", type=Path, help="Também grava o resumo neste arquivo.")
    return parser.parse_args()


def analyze(csv_path):

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    rows = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if len(rows) < 2:
        raise RuntimeError("São necessárias pelo menos duas amostras.")

    timestamps_ms = [int(r["epoch_ms"]) for r in rows]
    tapo_power_w = [float(r["tapo_power_w"]) for r in rows]
    voltage_v = [float(r["voltage_v"]) for r in rows]
    current_a = [float(r["current_a"]) for r in rows]
    cpu_percent = [float(r["cpu_percent"]) for r in rows]
    ram_percent = [float(r["ram_percent"]) for r in rows]
    gpu_util_percent = [float(r["gpu_util_percent"]) for r in rows]
    gpu_power_w = [float(r["gpu_power_w"]) for r in rows]
    gpu_memory_mib = [float(r["gpu_memory_mib"]) for r in rows]
    gpu_temp_c = [float(r["gpu_temp_c"]) for r in rows]

    intervals_s = [
        (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0
        for i in range(1, len(timestamps_ms))
    ]

    duration_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0

    tapo_energy_j = integrate_trapezoid(timestamps_ms, tapo_power_w)
    gpu_energy_j = integrate_trapezoid(timestamps_ms, gpu_power_w)

    tapo_energy_wh = tapo_energy_j / 3600.0
    gpu_energy_wh = gpu_energy_j / 3600.0

    print()
    print("=" * 64)
    print("RESUMO DA TELEMETRIA")
    print("=" * 64)

    print(f"Arquivo:                 {csv_path}")
    print(f"Início:                  {rows[0]['timestamp_local']}")
    print(f"Fim:                     {rows[-1]['timestamp_local']}")
    print(f"Amostras:                {len(rows)}")
    print(f"Duração real:            {duration_s:.3f} s")

    print()
    print("--- Intervalo de amostragem ---")
    print(f"Médio:                    {mean(intervals_s):.3f} s")
    print(f"Mediano:                  {statistics.median(intervals_s):.3f} s")
    print(f"Mínimo:                   {min(intervals_s):.3f} s")
    print(f"Máximo:                   {max(intervals_s):.3f} s")

    print()
    print("--- Tapo P110 / sistema completo ---")
    print(f"Potência média:           {mean(tapo_power_w):.3f} W")
    print(f"Potência mediana:         {statistics.median(tapo_power_w):.3f} W")
    print(f"Potência mínima:          {min(tapo_power_w):.3f} W")
    print(f"Potência máxima:          {max(tapo_power_w):.3f} W")
    print(f"Desvio-padrão:            {stdev(tapo_power_w):.3f} W")
    print(f"Tensão média:             {mean(voltage_v):.3f} V")
    print(f"Corrente média:           {mean(current_a):.3f} A")
    print(f"Energia integrada:        {tapo_energy_j:.3f} J")
    print(f"Energia integrada:        {tapo_energy_j / 1000:.6f} kJ")
    print(f"Energia integrada:        {tapo_energy_wh:.6f} Wh")

    print()
    print("--- CPU / memória ---")
    print(f"CPU média:                {mean(cpu_percent):.3f} %")
    print(f"RAM média:                {mean(ram_percent):.3f} %")

    print()
    print("--- RTX / NVIDIA-SMI ---")
    print(f"Utilização GPU média:     {mean(gpu_util_percent):.3f} %")
    print(f"Potência GPU média:       {mean(gpu_power_w):.3f} W")
    print(f"Potência GPU máxima:      {max(gpu_power_w):.3f} W")
    print(f"Memória GPU média:        {mean(gpu_memory_mib):.3f} MiB")
    print(f"Temperatura GPU média:    {mean(gpu_temp_c):.3f} °C")
    print(f"Energia GPU integrada:    {gpu_energy_j:.3f} J")
    print(f"Energia GPU integrada:    {gpu_energy_j / 1000:.6f} kJ")
    print(f"Energia GPU integrada:    {gpu_energy_wh:.6f} Wh")

    print()
    print("=" * 64)


def main():
    args = parse_args()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as output, contextlib.redirect_stdout(output):
            analyze(args.csv)
        print(args.output.read_text(), end="")
    else:
        analyze(args.csv)


if __name__ == "__main__":
    main()
