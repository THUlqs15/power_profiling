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
