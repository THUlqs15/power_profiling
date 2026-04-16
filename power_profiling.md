# Power Profiling: Reproducible Workflow

_Generated: 2026-04-17 01:32:24_

## Overview

This document records the exact steps, scripts, and environment used to produce `power_result.md`.

## Environment

```
Conda env  : myvllm
vLLM path  : /home/ubuntu/lqs/vllm
Model path : /home/ubuntu/lqs/LLM_model
GPU        : NVIDIA A800 SXM4 80GB
Mem clock  : 1593 MHz (hardware-fixed, HBM2e)
```

## Sweep Configuration

| Parameter | Value |
|-----------|-------|
| Prompt length | 1024 tokens |
| max_tokens | 1 (prefill isolation) |
| Warm-up duration | 60 s |
| Measurement duration | 120 s |
| Cooldown between steps | 30 s |
| SM clock range | 210-1410 MHz |
| Number of frequency points | 18 |

## Step-by-Step Reproduction

### Step 1: Activate environment

```bash
conda activate myvllm
```

### Step 2: Generate sweep clock list

```bash
nvidia-smi --query-supported-clocks=gr --format=csv,noheader,nounits | sort -n > supported_sm_clocks.txt
python -c "
import numpy as np
clocks=sorted(int(l) for l in open('supported_sm_clocks.txt') if l.strip())
idx=np.linspace(0,len(clocks)-1,18,dtype=int)
sel=sorted(set([clocks[i] for i in idx]+[clocks[0],clocks[-1]]))
open('sweep_clocks.txt','w').write('\n'.join(map(str,sel)))
print(sel)"
```

### Step 3: Verify clock control (requires root)

```bash
sudo bash set_sm_clock.sh 210
nvidia-smi --query-gpu=clocks.sm --format=csv,noheader
sudo bash set_sm_clock.sh reset
```

### Step 4: Calibrate saturation QPS

```bash
# Start vLLM, then increase --qps until GPU utilization >= 95%
watch -n1 "nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm --format=csv,noheader"
```

### Step 5: Run full sweep

```bash
sudo bash run_sweep.sh
```

### Step 6: Fit model and generate reports

```bash
python fit_power_model.py --results-dir ./profiling_results
```

## Key Design Decisions

**Why cubic polynomial?**
CMOS DVFS theory: P_dynamic is proportional to V^2*f, and V scales linearly with f, giving P proportional to f^3. The cubic form is chosen from physical prior; coefficients are fitted from data.

**Why fix memory clock?**
To isolate SM-frequency effects on power. On A800 SXM4, the HBM2e memory clock is hardware-fixed at 1593 MHz, so no explicit pinning command is needed.

**Why 1024-token prompts?**
Long enough to be firmly compute-bound (prefill FLOPs proportional to n^2), consistent with GreenLLM baseline, and fits within context limits.

**Why max_tokens=1?**
Terminates generation after a single decode step, ensuring the GPU spends >99% of time in the prefill (compute-bound) phase.

**Why 120 s measurement window?**
NVML power sensor refreshes every ~10-50 ms; 120 s gives ~2400-12000 samples, making the mean robust to transient spikes.

**Why k0 != P_idle?**
k0 is the polynomial's mathematical intercept (extrapolated to f=0) and absorbs constant non-SM power (HBM controllers, NVLink, board). P_idle must be measured separately with no active CUDA context.

## Measuring P_idle Separately

```bash
# With no GPU workload running:
nvidia-smi --query-gpu=power.draw --format=csv,noheader
# Or via Python:
python -c "
import pynvml, time
pynvml.nvmlInit()
h=pynvml.nvmlDeviceGetHandleByIndex(0)
readings=[pynvml.nvmlDeviceGetPowerUsage(h)/1000 for _ in range(30) if not time.sleep(1)]
print(f'P_idle = {sum(readings)/len(readings):.1f} W')"
```

## Fitted Results Summary

```
P(f) = 2.6770e-07*f^3 + -5.3135e-04*f^2 + 4.4484e-01*f + 1.8397e+01
R^2 = 0.9845   MAPE = 3.78%
```

## Limitations and Caveats

- NVML reports whole-card power (SM + HBM + NVLink + board); SM-only power is not directly observable.
- The model is calibrated at a single prompt length (1024 tokens). If prefill FLOP density changes significantly (e.g., very short prompts), a separate calibration per length class may be warranted.
- Power readings have +/-5-10 W sensor accuracy; MAPE < 2% is a realistic target.
- The model assumes the GPU is always at the locked SM frequency. In practice, thermal throttling can temporarily lower the actual clock below the locked value; monitor `clocks.sm` in the raw CSV to detect this.
- This calibration must be re-run if the model, GPU, or driver version changes.
