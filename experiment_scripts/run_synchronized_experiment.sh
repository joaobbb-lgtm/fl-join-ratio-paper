#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
FL_PYTHON="${FL_PYTHON:-$CONDA_ROOT/envs/fl-dissertacao/bin/python}"
TELEMETRY_PYTHON="${TELEMETRY_PYTHON:-$CONDA_ROOT/envs/telemetria/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiments}"
KASA_ENV_FILE="${KASA_ENV_FILE:-$HOME/.config/fl-telemetry/kasa.env}"

if [[ -f "$KASA_ENV_FILE" ]]; then
    # Arquivo local criado na preparação da telemetria; nunca é copiado aos logs.
    source "$KASA_ENV_FILE"
fi

ALGORITHM="${ALGORITHM:-FedAvg}"
DATASET="${DATASET:-FashionMNIST}"
MODEL="${MODEL:-CNN}"
NUM_CLASSES="${NUM_CLASSES:-10}"
DEVICE="${DEVICE:-cuda}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-50}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
JOIN_RATIO="${JOIN_RATIO:-0.1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
REPETITION="${REPETITION:-1}"
TELEMETRY_INTERVAL="${TELEMETRY_INTERVAL:-1.0}"
READY_TIMEOUT="${READY_TIMEOUT:-30}"

timestamp="$(date '+%Y%m%dT%H%M%S%z')"
safe_join_ratio="${JOIN_RATIO//./p}"
EXPERIMENT_ID="${EXPERIMENT_ID:-${timestamp}_${ALGORITHM}_${DATASET}_jr${safe_join_ratio}_rep${REPETITION}}"
EXPERIMENT_DIR="$OUTPUT_ROOT/$EXPERIMENT_ID"
TELEMETRY_CSV="$EXPERIMENT_DIR/telemetry.csv"
TELEMETRY_LOG="$EXPERIMENT_DIR/telemetry.log"
TELEMETRY_READY="$EXPERIMENT_DIR/.telemetry_ready"
PFL_LOG="$EXPERIMENT_DIR/pfllib.log"
SUMMARY_LOG="$EXPERIMENT_DIR/telemetry_summary.txt"
METADATA_FILE="$EXPERIMENT_DIR/metadata.env"
PFLLIB_SYSTEM="$REPO_ROOT/code/PFLlib/system"

telemetry_pid=""
pfl_pid=""

iso_now() { date '+%Y-%m-%dT%H:%M:%S.%3N%:z'; }

append_metadata() {
    printf '%s=%q\n' "$1" "$2" >> "$METADATA_FILE"
}

stop_telemetry() {
    if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
        kill -TERM "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
    fi
    telemetry_pid=""
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "$pfl_pid" ]] && kill -0 "$pfl_pid" 2>/dev/null; then
        kill -TERM "$pfl_pid" 2>/dev/null || true
        wait "$pfl_pid" 2>/dev/null || true
    fi
    stop_telemetry
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

[[ -x "$FL_PYTHON" ]] || { echo "Python de fl-dissertacao não encontrado: $FL_PYTHON" >&2; exit 1; }
[[ -x "$TELEMETRY_PYTHON" ]] || { echo "Python de telemetria não encontrado: $TELEMETRY_PYTHON" >&2; exit 1; }
[[ -f "$PFLLIB_SYSTEM/main.py" ]] || { echo "PFLlib não encontrado: $PFLLIB_SYSTEM" >&2; exit 1; }
[[ -n "${KASA_USERNAME:-}" ]] || { echo "KASA_USERNAME não está definida neste terminal." >&2; exit 1; }
[[ -n "${KASA_PASSWORD:-}" ]] || { echo "KASA_PASSWORD não está definida neste terminal." >&2; exit 1; }
[[ ! -e "$EXPERIMENT_DIR" ]] || { echo "ID de experimento já existe: $EXPERIMENT_ID" >&2; exit 1; }

mkdir -p "$EXPERIMENT_DIR"
touch "$METADATA_FILE"
for pair in \
    "EXPERIMENT_ID=$EXPERIMENT_ID" "ALGORITHM=$ALGORITHM" "DATASET=$DATASET" \
    "MODEL=$MODEL" "DEVICE=$DEVICE" "GLOBAL_ROUNDS=$GLOBAL_ROUNDS" \
    "NUM_CLIENTS=$NUM_CLIENTS" "JOIN_RATIO=$JOIN_RATIO" "LOCAL_EPOCHS=$LOCAL_EPOCHS" \
    "BATCH_SIZE=$BATCH_SIZE" "LEARNING_RATE=$LEARNING_RATE" "REPETITION=$REPETITION" \
    "FL_PYTHON=$FL_PYTHON" "TELEMETRY_PYTHON=$TELEMETRY_PYTHON" \
    "KASA_ENV_FILE=$KASA_ENV_FILE" \
    "HOSTNAME=$(hostname)" "ORCHESTRATOR_START=$(iso_now)"; do
    append_metadata "${pair%%=*}" "${pair#*=}"
done

echo "[$(iso_now)] Iniciando telemetria: $EXPERIMENT_ID"
"$TELEMETRY_PYTHON" -u "$REPO_ROOT/telemetry/collect_idle.py" \
    --experiment-id "$EXPERIMENT_ID" \
    --output "$TELEMETRY_CSV" \
    --ready-file "$TELEMETRY_READY" \
    --interval "$TELEMETRY_INTERVAL" \
    --duration 0 > "$TELEMETRY_LOG" 2>&1 &
telemetry_pid=$!

deadline=$((SECONDS + READY_TIMEOUT))
while [[ ! -e "$TELEMETRY_READY" ]]; do
    if ! kill -0 "$telemetry_pid" 2>/dev/null; then
        wait "$telemetry_pid" || true
        echo "A telemetria terminou antes da primeira amostra. Consulte $TELEMETRY_LOG" >&2
        exit 1
    fi
    (( SECONDS < deadline )) || { echo "Tempo esgotado aguardando a telemetria." >&2; exit 1; }
    sleep 0.2
done

append_metadata TELEMETRY_READY "$(iso_now)"
append_metadata PFL_START "$(iso_now)"
echo "[$(iso_now)] Telemetria pronta; iniciando PFLlib"

save_name="${DATASET}_${ALGORITHM}_${EXPERIMENT_ID}"
(
    cd "$PFLLIB_SYSTEM"
    "$FL_PYTHON" -u main.py \
        -dev "$DEVICE" -data "$DATASET" -m "$MODEL" \
    -ncl "$NUM_CLASSES" -algo "$ALGORITHM" \
        -gr "$GLOBAL_ROUNDS" -nc "$NUM_CLIENTS" -jr "$JOIN_RATIO" \
        -lbs "$BATCH_SIZE" -ls "$LOCAL_EPOCHS" -lr "$LEARNING_RATE" \
        --save_folder_name "$save_name"
) > "$PFL_LOG" 2>&1 &
pfl_pid=$!

set +e
wait "$pfl_pid"
pfl_exit_code=$?
set -e
pfl_pid=""
append_metadata PFL_END "$(iso_now)"
append_metadata PFL_EXIT_CODE "$pfl_exit_code"

stop_telemetry
append_metadata TELEMETRY_END "$(iso_now)"
rm -f "$TELEMETRY_READY"

if [[ -s "$TELEMETRY_CSV" ]]; then
    "$TELEMETRY_PYTHON" "$REPO_ROOT/telemetry/analyze_telemetry.py" \
        "$TELEMETRY_CSV" --output "$SUMMARY_LOG"
fi

append_metadata ORCHESTRATOR_END "$(iso_now)"
echo "[$(iso_now)] Experimento concluído (PFLlib exit=$pfl_exit_code): $EXPERIMENT_DIR"
exit "$pfl_exit_code"
