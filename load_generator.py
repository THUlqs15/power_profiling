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
    Rule of thumb: 1 token ~ 4 characters for English text."""
    chars = string.ascii_lowercase + " "
    raw = "".join(random.choices(chars, k=length * 4))
    return raw

def send_request(url: str, prompt: str, model_name: str, session: requests.Session):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1,             # generate exactly 1 token -> isolates prefill
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
        t = threading.Thread(target=send_request,
                             args=(url, prompt, args.model_name, session),
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
    parser.add_argument("--model-name", type=str, default="default")
    run(parser.parse_args())
