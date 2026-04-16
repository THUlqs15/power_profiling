# GPU Power Profiling Instruction
## Fitting P(f) = k₃f³ + k₂f² + k₁f + k₀ for LLM Prefill on a Single GPU

---

## 0. Background and Goals

This instruction reproduces the power-vs-SM-frequency profiling methodology from GreenLLM (Liu et al., 2025), adapted to your server environment. The goal is to fit a cubic polynomial

```
P(f) = k₃·f³ + k₂·f² + k₁·f + k₀
```

where `f` is the SM clock frequency (MHz) and `P(f)` is the GPU active power (W) during saturated prefill. The fitted model is later used to solve the SLO-aware energy optimization problem for the prefill stage.

**Two deliverables are produced at the end:**

| File | Contents |
|---|---|
| `power_result.md` | Fitted coefficients k₀–k₃, R², MAPE, raw data table, residual plot |
| `power_profiling.md` | Fully reproducible workflow, all scripts, environment notes |

---

## 1. Memory Frequency: Do Nothing

You mentioned setting memory frequency to 1593 MHz. **No action is needed.**

On A100 SXM4 (80 GB, HBM2e), 1593 MHz is the hardware-fixed memory clock. NVIDIA does not expose memory clock control on data-center Ampere SKUs: `nvmlDeviceSetMemoryLockedClocks` returns `NVML_SUCCESS` silently but has no effect. You can verify this:

```bash
nvidia-smi --query-gpu=clocks.mem --format=csv,noheader
# Expected output: 1593 MHz  (always, regardless of load)
```

This is actually ideal: memory clock isolation is guaranteed by hardware, so no software pinning step is required, and the measured `P(f)` reflects SM-frequency effects only.

---

## 2. Environment Prerequisites

```bash
# Activate the conda environment used for all experiments
conda activate myvllm

# Confirm vLLM and pynvml are present
python -c "import vllm; print(vllm.__version__)"
python -c "import pynvml; print('pynvml ok')"

# If pynvml is missing:
pip install pynvml --quiet

# Paths used throughout
VLLM_DIR=/home/ubuntu/lqs/vllm
MODEL_DIR=/home/ubuntu/lqs/LLM_model
```

---

## 3. vLLM Instrumentation (Non-Invasive)

To keep normal vLLM usage unaffected, the workload-generation script runs **outside** vLLM internals as a separate client process. No changes to vLLM source are required for the basic profiling workflow.

However, if you later want to embed profiling hooks (e.g., to log per-request SM frequency from inside the engine), the recommended pattern is:

```python
# Inside vllm/engine/async_llm_engine.py or a wrapper script
import os
PROFILING_ENABLED = os.environ.get("GREENLLM_PROFILING", "0") == "1"

if PROFILING_ENABLED:
    # insert NVML power/clock sampling here
    pass
```

Set `GREENLLM_PROFILING=1` only during experiments; normal launches are unaffected.

---

## 4. Hardware Constraints to Check First

Before starting the sweep, collect the list of valid SM clock values for your GPU. NVIDIA only accepts discrete supported clock values; arbitrary integers are rejected.

```bash
# List all supported SM (graphics) clocks at the default memory clock
nvidia-smi --query-supported-clocks=gr --format=csv,noheader,nounits \
  | sort -n | tee supported_sm_clocks.txt

# Typical A100 range: ~210 MHz to 1410 MHz, ~30 discrete steps
head -5 supported_sm_clocks.txt   # lowest clocks
tail -5 supported_sm_clocks.txt   # highest clocks
```

Use **only values from this list** when calling `nvidia-smi --lock-gpu-clocks` or NVML. Using unsupported values causes the command to fail silently or be rounded.

**Required permissions:** SM clock control requires root or a user with `CAP_SYS_ADMIN`. Verify:

```bash
sudo nvidia-smi --lock-gpu-clocks=210,210   # test with lowest clock
sudo nvidia-smi --reset-gpu-clocks          # restore immediately
```

If this fails, ask your system administrator to run the profiling script as root, or use `sudo` throughout.

---

## 5. Profiling Design

### 5.1 Saturation Strategy

The goal is to keep the GPU SM utilization at ~100% throughout each measurement interval so that measured power reflects the true active power at frequency `f`, not a mix of active and idle states.

**Workload parameters (prefill microbenchmark):**

| Parameter | Value | Rationale |
|---|---|---|
| Prompt length | 1024 tokens (fixed) | Matches GreenLLM; long enough to be compute-bound |
| Request rate | High enough to saturate (tune empirically, see §5.2) | Keeps SM busy >95% |
| Output tokens | 1 per request | Terminates decode immediately; isolates prefill |
| Batch size | vLLM default (continuous batching) | No manual tuning needed |
| Warm-up duration | 60 s | Allows clocks and thermals to stabilize |
| Measurement window | 120 s | Average over this window to reduce noise |
| Cooldown between steps | 30 s | Lets power settle before next frequency point |

### 5.2 Finding the Saturation Request Rate

Before the main sweep, run a brief calibration to find the minimum QPS that keeps GPU utilization ≥ 95%:

```bash
# Start vLLM server (one terminal)
conda activate myvllm
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_DIR \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 \
    --port 8000 &

# In another terminal: ramp QPS and watch utilization
watch -n1 "nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm \
           --format=csv,noheader"
```

Use the `load_generator.py` script (§6.2) with increasing `--qps` until `utilization.gpu` stabilizes at ≥ 95%. Record this as `QPS_SAT`. A typical value for a 14B dense model on A100 is 20–40 QPS at 1024-token prompts.

### 5.3 Frequency Sweep Points

From `supported_sm_clocks.txt`, select a representative subset spanning the full range:

```python
# Example selection logic (adapt based on actual supported clocks)
import numpy as np

with open("supported_sm_clocks.txt") as f:
    all_clocks = sorted(int(l.strip()) for l in f if l.strip())

# Select ~15–20 evenly spaced points + always include min and max
indices = np.linspace(0, len(all_clocks) - 1, 18, dtype=int)
sweep_clocks = sorted(set([all_clocks[i] for i in indices]
                          + [all_clocks[0], all_clocks[-1]]))
print(sweep_clocks)
```

Typical result for A100: `[210, 285, 360, 435, 510, 585, 660, 735, 810, 885, 960, 1035, 1110, 1185, 1260, 1335, 1410]` MHz (~17 points).

---

## 6. Scripts

### 6.1 `set_sm_clock.sh` — Clock Control Helper

```bash
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
    nvidia-smi --auto-boost-default=0 -i ${GPU_ID}
    echo "[clock] SM clock locked to ${FREQ} MHz on GPU ${GPU_ID}"
fi
```

### 6.2 `load_generator.py` — Saturating Prefill Workload

```python
#!/usr/bin/env python3
"""
Sends fixed-length prefill requests to a running vLLM OpenAI-compatible server.
Designed to saturate GPU SM during prefill-power profiling.

Usage:
    python load_generator.py --qps 30 --duration 200 --prompt-len 1024
"""
import argparse
import time
import threading
import requests
import random
import string

def make_prompt(length: int) -> str:
    """Generate a dummy prompt of approximately `length` tokens.
    Rule of thumb: 1 token ≈ 4 characters for English text."""
    chars = string.ascii_lowercase + " "
    raw = "".join(random.choices(chars, k=length * 4))
    return raw

def send_request(url: str, prompt: str, session: requests.Session):
    payload = {
        "model": "default",          # vLLM uses the loaded model name
        "prompt": prompt,
        "max_tokens": 1,             # generate exactly 1 token → isolates prefill
        "temperature": 0.0,
    }
    try:
        session.post(url, json=payload, timeout=30)
    except Exception:
        pass  # ignore errors during saturation run

def run(args):
    url = f"http://localhost:{args.port}/v1/completions"
    prompt = make_prompt(args.prompt_len)
    interval = 1.0 / args.qps
    session = requests.Session()

    deadline = time.time() + args.duration
    while time.time() < deadline:
        t = threading.Thread(target=send_request, args=(url, prompt, session),
                             daemon=True)
        t.start()
        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", type=float, default=30)
    parser.add_argument("--duration", type=float, default=200,
                        help="Total run duration in seconds")
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--port", type=int, default=8000)
    run(parser.parse_args())
```

### 6.3 `power_monitor.py` — NVML Power Sampler

```python
#!/usr/bin/env python3
"""
Samples GPU power and SM clock via NVML every 100 ms.
Writes a CSV: timestamp_s, power_mw, sm_clock_mhz

Usage:
    python power_monitor.py --gpu 0 --output power_raw.csv &
    MONITOR_PID=$!
    # ... run workload ...
    kill $MONITOR_PID
"""
import argparse
import csv
import time
import pynvml

def run(args):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "power_mw", "sm_clock_mhz"])
        while True:
            ts = time.time()
            pwr = pynvml.nvmlDeviceGetPowerUsage(handle)        # mW
            clk = pynvml.nvmlDeviceGetClockInfo(
                      handle, pynvml.NVML_CLOCK_SM)             # MHz
            writer.writerow([f"{ts:.3f}", pwr, clk])
            f.flush()
            time.sleep(args.interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="power_raw.csv")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="Sampling interval in seconds")
    run(parser.parse_args())
```

### 6.4 `run_sweep.sh` — Main Orchestration Script

```bash
#!/bin/bash
# Full SM-frequency sweep for prefill power profiling.
# Run as root (or with sudo) for clock control.
#
# Usage: sudo bash run_sweep.sh

set -e
conda activate myvllm

GPU_ID=0
PORT=8000
QPS_SAT=30           # set to your calibrated saturation QPS (see §5.2)
PROMPT_LEN=1024
WARMUP_S=60
MEASURE_S=120
COOLDOWN_S=30
MODEL_DIR=/home/ubuntu/lqs/LLM_model
RESULTS_DIR=./profiling_results
mkdir -p "$RESULTS_DIR"

# ── Step 1: Read sweep frequencies ────────────────────────────────────────────
mapfile -t CLOCKS < sweep_clocks.txt   # one MHz value per line
echo "[sweep] Frequencies to test: ${CLOCKS[*]}"

# ── Step 2: Start vLLM server ────────────────────────────────────────────────
echo "[sweep] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" &
VLLM_PID=$!
sleep 30   # wait for server to be ready
echo "[sweep] vLLM server PID: $VLLM_PID"

# ── Step 3: Sweep ─────────────────────────────────────────────────────────────
for FREQ in "${CLOCKS[@]}"; do
    echo "========================================"
    echo "[sweep] Testing SM clock = ${FREQ} MHz"

    # Lock SM clock
    bash set_sm_clock.sh "$FREQ"
    sleep 5   # let the clock settle

    # Start power monitor
    OUT_CSV="${RESULTS_DIR}/freq_${FREQ}.csv"
    python power_monitor.py --gpu "$GPU_ID" --output "$OUT_CSV" &
    MONITOR_PID=$!

    # Warm-up phase (not recorded separately; monitor runs throughout)
    echo "[sweep]   Warming up for ${WARMUP_S}s..."
    python load_generator.py --qps "$QPS_SAT" --duration "$WARMUP_S" \
        --prompt-len "$PROMPT_LEN" --port "$PORT"

    # Mark measurement start time
    MEASURE_START=$(date +%s%3N)   # ms since epoch

    # Measurement phase
    echo "[sweep]   Measuring for ${MEASURE_S}s..."
    python load_generator.py --qps "$QPS_SAT" --duration "$MEASURE_S" \
        --prompt-len "$PROMPT_LEN" --port "$PORT"

    MEASURE_END=$(date +%s%3N)

    # Stop power monitor
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true

    # Save measurement window timestamps for this frequency
    echo "${FREQ},${MEASURE_START},${MEASURE_END}" \
        >> "${RESULTS_DIR}/windows.csv"

    echo "[sweep]   Cooling down for ${COOLDOWN_S}s..."
    bash set_sm_clock.sh reset
    sleep "$COOLDOWN_S"
done

# ── Step 4: Teardown ──────────────────────────────────────────────────────────
kill "$VLLM_PID" 2>/dev/null || true
echo "[sweep] Done. Raw CSVs in ${RESULTS_DIR}/"
```

### 6.5 `fit_power_model.py` — Regression and Report Generator

```python
#!/usr/bin/env python3
"""
Reads per-frequency CSV files, computes mean power in the measurement window,
fits P(f) = k3*f^3 + k2*f^2 + k1*f + k0, reports coefficients + MAPE,
and writes power_result.md and power_profiling.md.

Usage:
    python fit_power_model.py --results-dir ./profiling_results
"""
import argparse
import csv
import os
import math
import numpy as np
from datetime import datetime

# ── Data loading ──────────────────────────────────────────────────────────────

def load_windows(results_dir):
    """Returns dict: freq_mhz -> (start_ms, end_ms)"""
    windows = {}
    path = os.path.join(results_dir, "windows.csv")
    with open(path) as f:
        for row in csv.reader(f):
            freq, start, end = int(row[0]), int(row[1]), int(row[2])
            windows[freq] = (start, end)
    return windows

def mean_power_in_window(csv_path, start_ms, end_ms):
    """Returns mean power (W) during [start_ms, end_ms]."""
    powers = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_ms = float(row["timestamp_s"]) * 1000
            if start_ms <= ts_ms <= end_ms:
                powers.append(float(row["power_mw"]) / 1000.0)  # mW → W
    if not powers:
        raise ValueError(f"No samples in window for {csv_path}")
    return float(np.mean(powers)), float(np.std(powers))

# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_cubic(freqs, powers):
    """Least-squares fit of cubic polynomial. Returns coefficients [k0,k1,k2,k3]."""
    f = np.array(freqs, dtype=float)
    p = np.array(powers, dtype=float)
    # np.polyfit returns highest-degree first: [k3, k2, k1, k0]
    coeffs = np.polyfit(f, p, deg=3)
    return coeffs  # [k3, k2, k1, k0]

def r_squared(freqs, powers, coeffs):
    f = np.array(freqs, dtype=float)
    p = np.array(powers, dtype=float)
    p_hat = np.polyval(coeffs, f)
    ss_res = np.sum((p - p_hat) ** 2)
    ss_tot = np.sum((p - np.mean(p)) ** 2)
    return 1.0 - ss_res / ss_tot

def mape(freqs, powers, coeffs):
    f = np.array(freqs, dtype=float)
    p = np.array(powers, dtype=float)
    p_hat = np.polyval(coeffs, f)
    return float(np.mean(np.abs((p - p_hat) / p)) * 100)

# ── Report writers ────────────────────────────────────────────────────────────

def write_power_result(freqs, powers, stds, coeffs, r2, mape_val):
    k3, k2, k1, k0 = coeffs
    lines = []
    lines.append("# Power Model Fitting Results\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

    lines.append("## Fitted Model\n\n")
    lines.append("```\n")
    lines.append("P(f) = k3·f³ + k2·f² + k1·f + k0\n")
    lines.append(f"\nk3 = {k3:.6e}\n")
    lines.append(f"k2 = {k2:.6e}\n")
    lines.append(f"k1 = {k1:.6e}\n")
    lines.append(f"k0 = {k0:.6e}\n")
    lines.append("```\n\n")

    lines.append("## Goodness of Fit\n\n")
    lines.append(f"| Metric | Value |\n|--------|-------|\n")
    lines.append(f"| R²     | {r2:.4f} |\n")
    lines.append(f"| MAPE   | {mape_val:.2f}% |\n\n")

    lines.append("## Raw Measurements\n\n")
    lines.append("| SM Clock (MHz) | Mean Power (W) | Std (W) | Fitted P(f) (W) | Error (%) |\n")
    lines.append("|---------------|---------------|---------|-----------------|----------|\n")
    for freq, pwr, std in zip(freqs, powers, stds):
        fitted = float(np.polyval(coeffs, freq))
        err = abs(pwr - fitted) / pwr * 100
        lines.append(f"| {freq:>14} | {pwr:>13.2f} | {std:>7.2f} | {fitted:>15.2f} | {err:>8.2f} |\n")

    lines.append("\n## Notes\n\n")
    lines.append("- Power measured via NVML `nvmlDeviceGetPowerUsage` (whole-card, mW resolution).\n")
    lines.append("- Memory clock fixed at hardware default (1593 MHz for A100 SXM4 80GB; not software-controllable).\n")
    lines.append("- `k0` is the polynomial intercept and does **not** equal `P_idle` (measured separately).\n")
    lines.append("- Model valid only under saturated prefill load at the profiled prompt length (1024 tokens).\n")

    with open("power_result.md", "w") as f:
        f.writelines(lines)
    print("[fit] Wrote power_result.md")

def write_power_profiling(freqs, powers, coeffs, r2, mape_val):
    k3, k2, k1, k0 = coeffs
    lines = []
    lines.append("# Power Profiling: Reproducible Workflow\n\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

    lines.append("## Overview\n\n")
    lines.append("This document records the exact steps, scripts, and environment used to produce `power_result.md`.\n\n")

    lines.append("## Environment\n\n")
    lines.append("```\n")
    lines.append("Conda env  : myvllm\n")
    lines.append("vLLM path  : /home/ubuntu/lqs/vllm\n")
    lines.append("Model path : /home/ubuntu/lqs/LLM_model\n")
    lines.append("GPU        : NVIDIA A100 SXM4 80GB\n")
    lines.append("Mem clock  : 1593 MHz (hardware-fixed, HBM2e)\n")
    lines.append("```\n\n")

    lines.append("## Sweep Configuration\n\n")
    lines.append(f"| Parameter | Value |\n|-----------|-------|\n")
    lines.append(f"| Prompt length | 1024 tokens |\n")
    lines.append(f"| max_tokens | 1 (prefill isolation) |\n")
    lines.append(f"| Warm-up duration | 60 s |\n")
    lines.append(f"| Measurement duration | 120 s |\n")
    lines.append(f"| Cooldown between steps | 30 s |\n")
    lines.append(f"| SM clock range | {min(freqs)}–{max(freqs)} MHz |\n")
    lines.append(f"| Number of frequency points | {len(freqs)} |\n\n")

    lines.append("## Step-by-Step Reproduction\n\n")
    steps = [
        ("Activate environment", "conda activate myvllm"),
        ("Generate sweep clock list",
         "nvidia-smi --query-supported-clocks=gr --format=csv,noheader,nounits "
         "| sort -n > supported_sm_clocks.txt\n"
         "python -c \"\nimport numpy as np\nclocks=sorted(int(l) for l in "
         "open('supported_sm_clocks.txt') if l.strip())\n"
         "idx=np.linspace(0,len(clocks)-1,18,dtype=int)\n"
         "sel=sorted(set([clocks[i] for i in idx]+[clocks[0],clocks[-1]]))\n"
         "open('sweep_clocks.txt','w').write('\\n'.join(map(str,sel)))\n"
         "print(sel)\""),
        ("Verify clock control (requires root)",
         "sudo bash set_sm_clock.sh 210\n"
         "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader\n"
         "sudo bash set_sm_clock.sh reset"),
        ("Calibrate saturation QPS",
         "# Start vLLM, then increase --qps until GPU utilization >= 95%\n"
         "watch -n1 \"nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm "
         "--format=csv,noheader\""),
        ("Run full sweep", "sudo bash run_sweep.sh"),
        ("Fit model and generate reports",
         "python fit_power_model.py --results-dir ./profiling_results"),
    ]
    for i, (title, cmd) in enumerate(steps, 1):
        lines.append(f"### Step {i}: {title}\n\n```bash\n{cmd}\n```\n\n")

    lines.append("## Key Design Decisions\n\n")
    decisions = [
        ("Why cubic polynomial?",
         "CMOS DVFS theory: P_dynamic ∝ V²·f, and V scales linearly with f, "
         "giving P ∝ f³. The cubic form is chosen from physical prior; "
         "coefficients are fitted from data."),
        ("Why fix memory clock?",
         "To isolate SM-frequency effects on power. On A100 SXM4, "
         "the HBM2e memory clock is hardware-fixed at 1593 MHz, "
         "so no explicit pinning command is needed."),
        ("Why 1024-token prompts?",
         "Long enough to be firmly compute-bound (prefill FLOPs ∝ n²), "
         "consistent with GreenLLM baseline, and fits within A100 context limits."),
        ("Why max_tokens=1?",
         "Terminates generation after a single decode step, ensuring the GPU "
         "spends >99% of time in the prefill (compute-bound) phase."),
        ("Why 120 s measurement window?",
         "NVML power sensor refreshes every ~10–50 ms; 120 s gives ~2400–12000 "
         "samples, making the mean robust to transient spikes."),
        ("Why k0 ≠ P_idle?",
         "k0 is the polynomial's mathematical intercept (extrapolated to f=0) "
         "and absorbs constant non-SM power (HBM controllers, NVLink, board). "
         "P_idle must be measured separately with no active CUDA context."),
    ]
    for q, a in decisions:
        lines.append(f"**{q}**\n{a}\n\n")

    lines.append("## Measuring P_idle Separately\n\n")
    lines.append("```bash\n")
    lines.append("# With no GPU workload running:\n")
    lines.append("nvidia-smi --query-gpu=power.draw --format=csv,noheader\n")
    lines.append("# Or via Python:\n")
    lines.append("python -c \"\nimport pynvml, time\npynvml.nvmlInit()\n"
                 "h=pynvml.nvmlDeviceGetHandleByIndex(0)\n"
                 "readings=[pynvml.nvmlDeviceGetPowerUsage(h)/1000 for _ in "
                 "range(30) if not time.sleep(1)]\n"
                 "print(f'P_idle = {sum(readings)/len(readings):.1f} W')\"\n")
    lines.append("```\n\n")

    lines.append("## Fitted Results Summary\n\n")
    lines.append(f"```\nP(f) = {k3:.4e}·f³ + {k2:.4e}·f² + {k1:.4e}·f + {k0:.4e}\n")
    lines.append(f"R² = {r2:.4f}   MAPE = {mape_val:.2f}%\n```\n\n")

    lines.append("## Limitations and Caveats\n\n")
    caveats = [
        "NVML reports whole-card power (SM + HBM + NVLink + board); "
        "SM-only power is not directly observable.",
        "The model is calibrated at a single prompt length (1024 tokens). "
        "If prefill FLOP density changes significantly (e.g., very short prompts), "
        "a separate calibration per length class may be warranted.",
        "Power readings have ±5–10 W sensor accuracy; MAPE < 2% is a realistic target.",
        "The model assumes the GPU is always at the locked SM frequency. "
        "In practice, thermal throttling can temporarily lower the actual clock "
        "below the locked value; monitor `clocks.sm` in the raw CSV to detect this.",
        "This calibration must be re-run if the model, GPU, or driver version changes.",
    ]
    for c in caveats:
        lines.append(f"- {c}\n")

    with open("power_profiling.md", "w") as f:
        f.writelines(lines)
    print("[fit] Wrote power_profiling.md")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./profiling_results")
    args = parser.parse_args()

    windows = load_windows(args.results_dir)
    freqs, powers, stds = [], [], []

    for freq in sorted(windows):
        csv_path = os.path.join(args.results_dir, f"freq_{freq}.csv")
        start_ms, end_ms = windows[freq]
        mean_w, std_w = mean_power_in_window(csv_path, start_ms, end_ms)
        freqs.append(freq)
        powers.append(mean_w)
        stds.append(std_w)
        print(f"  {freq:>6} MHz  →  {mean_w:.1f} ± {std_w:.1f} W")

    coeffs = fit_cubic(freqs, powers)
    r2 = r_squared(freqs, powers, coeffs)
    mape_val = mape(freqs, powers, coeffs)

    k3, k2, k1, k0 = coeffs
    print(f"\nFitted: P(f) = {k3:.4e}·f³ + {k2:.4e}·f² + {k1:.4e}·f + {k0:.4e}")
    print(f"R² = {r2:.4f},  MAPE = {mape_val:.2f}%")

    write_power_result(freqs, powers, stds, coeffs, r2, mape_val)
    write_power_profiling(freqs, powers, coeffs, r2, mape_val)

if __name__ == "__main__":
    main()
```

---

## 7. Execution Order Summary

```
1. conda activate myvllm
2. nvidia-smi --query-supported-clocks=gr ... > supported_sm_clocks.txt
3. python (select sweep points) → sweep_clocks.txt
4. sudo bash set_sm_clock.sh 210   # permission check
5. Start vLLM, calibrate QPS_SAT, update run_sweep.sh
6. sudo bash run_sweep.sh          # ~(17 points × (60+120+30)s) ≈ 1.7 hours
7. python fit_power_model.py --results-dir ./profiling_results
   → power_result.md
   → power_profiling.md
```

---

## 8. Quality Checks

After the sweep, before fitting, verify the raw data:

```bash
# Check that measured SM clock matches the locked value (detect thermal throttling)
for f in profiling_results/freq_*.csv; do
  echo -n "$f: "
  awk -F',' 'NR>1{sum+=$3; n++} END{printf "mean SM clock = %.0f MHz\n", sum/n}' "$f"
done

# Check power is monotonically increasing with frequency in the high-frequency range
# (rough sanity: should see clear upward trend above ~800 MHz)
awk -F',' 'NR>1{print $1, $2}' profiling_results/windows.csv  # spot check
```

If any frequency point shows SM clock significantly below the locked value (>50 MHz deviation), the GPU was thermally throttling during that measurement — discard that point and re-run with a longer cooldown.

---

## 9. Expected Results

Based on GreenLLM results on A100 SXM4 40GB (similar architecture):

- Power range: ~100 W (idle) to ~380 W (peak SM load at 1410 MHz)
- U-shaped energy curve minimum around 950–1050 MHz
- R² ≥ 0.99 is achievable with a cubic polynomial
- MAPE < 2% is a realistic target

The 80 GB variant has a higher TDP (400 W) and slightly different power characteristics; expect similar curve shape but potentially higher absolute values.
