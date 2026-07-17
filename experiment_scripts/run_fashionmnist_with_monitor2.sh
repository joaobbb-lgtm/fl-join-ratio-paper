#!/bin/bash
# =============================================================================
# Experimentos FL - FashionMNIST - GPU (WSL + MX230)
# Saída externa: ~/research/experiments/fashionmnist_gpu/<algoritmo>/
# Resultados internos do PFLlib: ~/research/projects/PFLlib/results/
#
# GPU log com timestamp explícito via nvidia-smi:
# timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,
# temperature.gpu,power.draw
# =============================================================================

set -u

# -----------------------------------------------------------------------------
# Configuração principal
# -----------------------------------------------------------------------------

PFLLIB_DIR="$HOME/research/projects/PFLlib/system"
OUTPUT_ROOT="$HOME/research/experiments/fashionmnist_gpu"
PYTHON_BIN="$HOME/fl_env/bin/python"

DATASET="FashionMNIST"
MODEL="CNN"
DEVICE="cuda"

# Escolha UM algoritmo por vez: FedAvg | FedProx | SCAFFOLD
ALGO="SCAFFOLD"

NUM_CLIENTS=100
GLOBAL_ROUNDS=50
LOCAL_EPOCHS=5
BATCH_SIZE=10
LOCAL_LEARNING_RATE=0.005

JOIN_RATIOS=(0.75)
REPETITIONS=1

PIDSTAT_INTERVAL=1
GPU_LOG_INTERVAL=1

# -----------------------------------------------------------------------------
# Caminhos derivados
# -----------------------------------------------------------------------------

OUTPUT_BASE="$OUTPUT_ROOT/$ALGO"
PROGRESS_FILE="$OUTPUT_BASE/progress.log"
ORCH_LOG="$OUTPUT_BASE/orchestrator.log"

# -----------------------------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------------------------

log() {
    mkdir -p "$OUTPUT_BASE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ORCH_LOG"
}

already_done() {
    local tag="$1"
    grep -qF "$tag" "$PROGRESS_FILE" 2>/dev/null
}

mark_done() {
    local tag="$1"
    echo "$tag" >> "$PROGRESS_FILE"
}

cleanup_children() {
    if [[ -n "${GPUSTAT_PID:-}" ]]; then
        kill "$GPUSTAT_PID" 2>/dev/null || true
        wait "$GPUSTAT_PID" 2>/dev/null || true
        unset GPUSTAT_PID
    fi

    if [[ -n "${PIDSTAT_PID:-}" ]]; then
        kill "$PIDSTAT_PID" 2>/dev/null || true
        wait "$PIDSTAT_PID" 2>/dev/null || true
        unset PIDSTAT_PID
    fi

    if [[ -n "${PY_PID:-}" ]]; then
        kill "$PY_PID" 2>/dev/null || true
        wait "$PY_PID" 2>/dev/null || true
        unset PY_PID
    fi
}

check_requirements() {
    [[ -x "$PYTHON_BIN" ]] || {
        echo "ERRO: Python do ambiente não encontrado em: $PYTHON_BIN"
        exit 1
    }

    command -v pidstat >/dev/null 2>&1 || {
        echo "ERRO: pidstat não encontrado. Instale o pacote sysstat."
        exit 1
    }

    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "ERRO: nvidia-smi não encontrado. GPU/driver não disponível neste ambiente."
        exit 1
    }

    [[ -d "$PFLLIB_DIR" ]] || {
        echo "ERRO: diretório não encontrado: $PFLLIB_DIR"
        exit 1
    }

    [[ -f "$PFLLIB_DIR/main.py" ]] || {
        echo "ERRO: main.py não encontrado em: $PFLLIB_DIR"
        exit 1
    }

    case "$ALGO" in
        FedAvg|FedProx|SCAFFOLD) ;;
        *)
            echo "ERRO: algoritmo inválido: $ALGO"
            echo "Use: FedAvg | FedProx | SCAFFOLD"
            exit 1
            ;;
    esac
}

trap 'log "Interrupção detectada. Encerrando processos filhos..."; cleanup_children; exit 130' INT TERM
trap 'cleanup_children' EXIT

# -----------------------------------------------------------------------------
# Setup inicial
# -----------------------------------------------------------------------------

check_requirements

mkdir -p "$OUTPUT_BASE"
touch "$PROGRESS_FILE"

TOTAL=$(( ${#JOIN_RATIOS[@]} * REPETITIONS ))
CURRENT=0

log "============================================================"
log "Iniciando experimentos FL — FashionMNIST (GPU)"
log "Algoritmo: $ALGO"
log "Device: $DEVICE"
log "Diretório PFLlib: $PFLLIB_DIR"
log "Saída externa: $OUTPUT_BASE"
log "Python: $PYTHON_BIN"
log "Total de execuções planejadas: $TOTAL"
log "============================================================"

# -----------------------------------------------------------------------------
# Loop principal
# -----------------------------------------------------------------------------

for JR in "${JOIN_RATIOS[@]}"; do
    for REP in $(seq 1 "$REPETITIONS"); do

        CURRENT=$((CURRENT + 1))

        # 0.75 -> 075
        JR_TAG=$(echo "$JR" | sed 's/\.//g')

        RERUN_LABEL="rep6_rerun_outlier"

        EXP_TAG="${ALGO}_GPU_JR${JR_TAG}_${RERUN_LABEL}"
        EXP_DIR="$OUTPUT_BASE/jr${JR_TAG}/rep6_rerun_outlier"

        mkdir -p "$EXP_DIR"

        RAW_LOG="$EXP_DIR/raw_log.txt"
        CPU_LOG="$EXP_DIR/cpu_usage.log"
        GPU_LOG="$EXP_DIR/gpu_usage.csv"
        META_LOG="$EXP_DIR/meta_info.txt"
        PID_FILE="$EXP_DIR/python.pid"
        PIDSTAT_PID_FILE="$EXP_DIR/pidstat.pid"
        GPUSTAT_PID_FILE="$EXP_DIR/gpustat.pid"

        SAVE_NAME="FashionMNIST_${ALGO}_gpu_jr${JR_TAG}_${RERUN_LABEL}"

        if already_done "$EXP_TAG"; then
            log "[$CURRENT/$TOTAL] SKIP (já concluído): $EXP_TAG"
            continue
        fi

        log "[$CURRENT/$TOTAL] Iniciando: $EXP_TAG"

        {
    echo "EXP_TAG=$EXP_TAG"
    echo "ALGORITHM=$ALGO"
    echo "DEVICE=$DEVICE"
    echo "JOIN_RATIO=$JR"
    echo "REPETITION=$REP"
    echo "DATASET=$DATASET"
    echo "MODEL=$MODEL"
    echo "NUM_CLIENTS=$NUM_CLIENTS"
    echo "GLOBAL_ROUNDS=$GLOBAL_ROUNDS"
    echo "LOCAL_EPOCHS=$LOCAL_EPOCHS"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "LOCAL_LEARNING_RATE=$LOCAL_LEARNING_RATE"
    echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "HOSTNAME=$(hostname)"
    echo "PYTORCH_VERSION=$("$PYTHON_BIN" -c 'import torch; print(torch.__version__)')"
    echo "CUDA_AVAILABLE=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.is_available())')"
    echo "GPU_NAME=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")')"
    echo "RERUN_OF=rep5"
    echo "RERUN_REASON=outlier_in_total_time_with_normal_accuracy"
    echo "RERUN_TYPE=targeted_validation"

} > "$META_LOG"

        cd "$PFLLIB_DIR" || {
            log "ERRO: falha ao acessar $PFLLIB_DIR"
            exit 1
        }

        "$PYTHON_BIN" -u main.py \
            -dev "$DEVICE" \
            -data "$DATASET" \
            -m "$MODEL" \
            -algo "$ALGO" \
            -gr "$GLOBAL_ROUNDS" \
            -nc "$NUM_CLIENTS" \
            -jr "$JR" \
            -lbs "$BATCH_SIZE" \
            -ls "$LOCAL_EPOCHS" \
            -lr "$LOCAL_LEARNING_RATE" \
            --save_folder_name "$SAVE_NAME" \
            > "$RAW_LOG" 2>&1 &

        PY_PID=$!
        echo "$PY_PID" > "$PID_FILE"

        log "[$CURRENT/$TOTAL] PID do experimento: $PY_PID"

        sleep 2

        if ! kill -0 "$PY_PID" 2>/dev/null; then
            log "[$CURRENT/$TOTAL] ERRO: processo morreu logo após iniciar: $EXP_TAG"
            echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')" >> "$META_LOG"
            echo "EXIT_CODE=999" >> "$META_LOG"
            unset PY_PID
            continue
        fi

        # ---------------------------------------------------------------------
        # Monitoramento de CPU
        # ---------------------------------------------------------------------
        pidstat -rud -h "$PIDSTAT_INTERVAL" -p "$PY_PID" > "$CPU_LOG" &
        PIDSTAT_PID=$!
        echo "$PIDSTAT_PID" > "$PIDSTAT_PID_FILE"

        # ---------------------------------------------------------------------
        # Monitoramento de GPU com timestamp explícito
        # A coluna "timestamp" será a primeira coluna do CSV.
        # ---------------------------------------------------------------------
        nvidia-smi \
            --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits \
            -l "$GPU_LOG_INTERVAL" \
            > "$GPU_LOG.tmp" &
        GPUSTAT_PID=$!
        echo "$GPUSTAT_PID" > "$GPUSTAT_PID_FILE"

        # Cabeçalho controlado manualmente para deixar explícito que há timestamp
        {
            echo "timestamp,gpu_name,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,temperature_gpu_c,power_draw_w"
            cat "$GPU_LOG.tmp"
        } > "$GPU_LOG" &
        GPU_MERGE_PID=$!

        wait "$PY_PID"
        EXIT_CODE=$?
        unset PY_PID

        if [[ -n "${PIDSTAT_PID:-}" ]]; then
            kill "$PIDSTAT_PID" 2>/dev/null || true
            wait "$PIDSTAT_PID" 2>/dev/null || true
            unset PIDSTAT_PID
        fi

        if [[ -n "${GPUSTAT_PID:-}" ]]; then
            kill "$GPUSTAT_PID" 2>/dev/null || true
            wait "$GPUSTAT_PID" 2>/dev/null || true
            unset GPUSTAT_PID
        fi

        if [[ -n "${GPU_MERGE_PID:-}" ]]; then
            wait "$GPU_MERGE_PID" 2>/dev/null || true
            unset GPU_MERGE_PID
        fi

        # Recria o CSV final de forma limpa
        {
            echo "timestamp,gpu_name,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,temperature_gpu_c,power_draw_w"
            cat "$GPU_LOG.tmp" 2>/dev/null
        } > "$GPU_LOG"

        rm -f "$GPU_LOG.tmp"

        echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')" >> "$META_LOG"
        echo "EXIT_CODE=$EXIT_CODE" >> "$META_LOG"

        if [[ $EXIT_CODE -eq 0 ]]; then
            log "[$CURRENT/$TOTAL] CONCLUÍDO: $EXP_TAG"
            mark_done "$EXP_TAG"
        else
            log "[$CURRENT/$TOTAL] ERRO (exit $EXIT_CODE): $EXP_TAG — verifique $RAW_LOG"
        fi

        sleep 3
    done
done

log "============================================================"
log "Experimentos concluídos para o algoritmo: $ALGO"
log "Resultados externos em: $OUTPUT_BASE"
log "============================================================"