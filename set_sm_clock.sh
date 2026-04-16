#!/bin/bash
# Usage: sudo bash set_sm_clock.sh <freq_mhz>
#        sudo bash set_sm_clock.sh reset

set -e
GPU_ID=${GPU_ID:-0}   # change if targeting a specific GPU

if [ "$1" == "reset" ]; then
    nvidia-smi --reset-gpu-clocks
    echo "[clock] Reset to default boost clocks"
else
    FREQ=$1
    # Lock SM clock to exactly FREQ MHz (min=max=FREQ)
    nvidia-smi --lock-gpu-clocks=${FREQ},${FREQ} -i ${GPU_ID}
    # Disable autoboost to prevent the driver from overriding the lock
    nvidia-smi --auto-boost-default=0 -i ${GPU_ID} 2>/dev/null || true
    echo "[clock] SM clock locked to ${FREQ} MHz on GPU ${GPU_ID}"
fi
