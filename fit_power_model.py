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

# -- Data loading --------------------------------------------------------------

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
                powers.append(float(row["power_mw"]) / 1000.0)  # mW -> W
    if not powers:
        raise ValueError(f"No samples in window for {csv_path}")
    return float(np.mean(powers)), float(np.std(powers))

# -- Fitting -------------------------------------------------------------------

def fit_cubic(freqs, powers):
    """Least-squares fit of cubic polynomial. Returns coefficients [k3,k2,k1,k0]."""
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

# -- Report writers ------------------------------------------------------------

def write_power_result(freqs, powers, stds, coeffs, r2, mape_val):
    k3, k2, k1, k0 = coeffs
    lines = []
    lines.append("# Power Model Fitting Results\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

    lines.append("## Fitted Model\n\n")
    lines.append("```\n")
    lines.append("P(f) = k3*f^3 + k2*f^2 + k1*f + k0\n")
    lines.append(f"\nk3 = {k3:.6e}\n")
    lines.append(f"k2 = {k2:.6e}\n")
    lines.append(f"k1 = {k1:.6e}\n")
    lines.append(f"k0 = {k0:.6e}\n")
    lines.append("```\n\n")

    lines.append("## Goodness of Fit\n\n")
    lines.append(f"| Metric | Value |\n|--------|-------|\n")
    lines.append(f"| R^2    | {r2:.4f} |\n")
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
    lines.append("- Memory clock fixed at hardware default (1593 MHz for A800 SXM4 80GB; not software-controllable).\n")
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
    lines.append("GPU        : NVIDIA A800 SXM4 80GB\n")
    lines.append("Mem clock  : 1593 MHz (hardware-fixed, HBM2e)\n")
    lines.append("```\n\n")

    lines.append("## Sweep Configuration\n\n")
    lines.append(f"| Parameter | Value |\n|-----------|-------|\n")
    lines.append(f"| Prompt length | 1024 tokens |\n")
    lines.append(f"| max_tokens | 1 (prefill isolation) |\n")
    lines.append(f"| Warm-up duration | 60 s |\n")
    lines.append(f"| Measurement duration | 120 s |\n")
    lines.append(f"| Cooldown between steps | 30 s |\n")
    lines.append(f"| SM clock range | {min(freqs)}-{max(freqs)} MHz |\n")
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
         "CMOS DVFS theory: P_dynamic is proportional to V^2*f, and V scales linearly with f, "
         "giving P proportional to f^3. The cubic form is chosen from physical prior; "
         "coefficients are fitted from data."),
        ("Why fix memory clock?",
         "To isolate SM-frequency effects on power. On A800 SXM4, "
         "the HBM2e memory clock is hardware-fixed at 1593 MHz, "
         "so no explicit pinning command is needed."),
        ("Why 1024-token prompts?",
         "Long enough to be firmly compute-bound (prefill FLOPs proportional to n^2), "
         "consistent with GreenLLM baseline, and fits within context limits."),
        ("Why max_tokens=1?",
         "Terminates generation after a single decode step, ensuring the GPU "
         "spends >99% of time in the prefill (compute-bound) phase."),
        ("Why 120 s measurement window?",
         "NVML power sensor refreshes every ~10-50 ms; 120 s gives ~2400-12000 "
         "samples, making the mean robust to transient spikes."),
        ("Why k0 != P_idle?",
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
    lines.append(f"```\nP(f) = {k3:.4e}*f^3 + {k2:.4e}*f^2 + {k1:.4e}*f + {k0:.4e}\n")
    lines.append(f"R^2 = {r2:.4f}   MAPE = {mape_val:.2f}%\n```\n\n")

    lines.append("## Limitations and Caveats\n\n")
    caveats = [
        "NVML reports whole-card power (SM + HBM + NVLink + board); "
        "SM-only power is not directly observable.",
        "The model is calibrated at a single prompt length (1024 tokens). "
        "If prefill FLOP density changes significantly (e.g., very short prompts), "
        "a separate calibration per length class may be warranted.",
        "Power readings have +/-5-10 W sensor accuracy; MAPE < 2% is a realistic target.",
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

# -- Main ---------------------------------------------------------------------

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
        print(f"  {freq:>6} MHz  ->  {mean_w:.1f} +/- {std_w:.1f} W")

    coeffs = fit_cubic(freqs, powers)
    r2 = r_squared(freqs, powers, coeffs)
    mape_val = mape(freqs, powers, coeffs)

    k3, k2, k1, k0 = coeffs
    print(f"\nFitted: P(f) = {k3:.4e}*f^3 + {k2:.4e}*f^2 + {k1:.4e}*f + {k0:.4e}")
    print(f"R^2 = {r2:.4f},  MAPE = {mape_val:.2f}%")

    write_power_result(freqs, powers, stds, coeffs, r2, mape_val)
    write_power_profiling(freqs, powers, coeffs, r2, mape_val)

if __name__ == "__main__":
    main()
