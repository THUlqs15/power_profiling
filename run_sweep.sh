#!/bin/bash
# Full SM-frequency sweep for prefill power profiling.
# Run as root (or with sudo) for clock control.
#
# Usage: sudo bash run_sweep.sh

set -e

# Source conda
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate myvllm

GPU_ID=0
PORT=8000
QPS_SAT=50           # calibrated: 100% GPU utilization at default clocks
PROMPT_LEN=1024
WARMUP_S=60
MEASURE_S=120
COOLDOWN_S=30
MODEL_DIR=/home/ubuntu/lqs/LLM_model
MODEL_NAME="default"
RESULTS_DIR=./profiling_results
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$RESULTS_DIR"

# -- Step 1: Read sweep frequencies --------------------------------------------
mapfile -t CLOCKS < "${SCRIPT_DIR}/sweep_clocks.txt"   # one MHz value per line
echo "[sweep] Frequencies to test: ${CLOCKS[*]}"

# -- Step 2: Start vLLM server ------------------------------------------------
echo "[sweep] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" &
VLLM_PID=$!

# Wait for server to be ready (check health endpoint)
echo "[sweep] Waiting for vLLM server to start (PID: $VLLM_PID)..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "[sweep] vLLM server is ready."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[sweep] ERROR: vLLM server exited unexpectedly."
        exit 1
    fi
    sleep 5
done

# -- Step 3: Sweep -------------------------------------------------------------
for FREQ in "${CLOCKS[@]}"; do
    echo "========================================"
    echo "[sweep] Testing SM clock = ${FREQ} MHz"

    # Lock SM clock
    bash "${SCRIPT_DIR}/set_sm_clock.sh" "$FREQ"
    sleep 5   # let the clock settle

    # Start power monitor
    OUT_CSV="${RESULTS_DIR}/freq_${FREQ}.csv"
    python "${SCRIPT_DIR}/power_monitor.py" --gpu "$GPU_ID" --output "$OUT_CSV" &
    MONITOR_PID=$!

    # Warm-up phase (not recorded separately; monitor runs throughout)
    echo "[sweep]   Warming up for ${WARMUP_S}s..."
    python "${SCRIPT_DIR}/load_generator.py" --qps "$QPS_SAT" --duration "$WARMUP_S" \
        --prompt-len "$PROMPT_LEN" --port "$PORT" --model-name "$MODEL_NAME"

    # Mark measurement start time
    MEASURE_START=$(date +%s%3N)   # ms since epoch

    # Measurement phase
    echo "[sweep]   Measuring for ${MEASURE_S}s..."
    python "${SCRIPT_DIR}/load_generator.py" --qps "$QPS_SAT" --duration "$MEASURE_S" \
        --prompt-len "$PROMPT_LEN" --port "$PORT" --model-name "$MODEL_NAME"

    MEASURE_END=$(date +%s%3N)

    # Stop power monitor
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true

    # Save measurement window timestamps for this frequency
    echo "${FREQ},${MEASURE_START},${MEASURE_END}" \
        >> "${RESULTS_DIR}/windows.csv"

    echo "[sweep]   Cooling down for ${COOLDOWN_S}s..."
    bash "${SCRIPT_DIR}/set_sm_clock.sh" reset
    sleep "$COOLDOWN_S"
done

# -- Step 4: Teardown ---------------------------------------------------------
kill "$VLLM_PID" 2>/dev/null || true
echo "[sweep] Done. Raw CSVs in ${RESULTS_DIR}/"
