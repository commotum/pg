#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


def last_float(pattern, text):
    vals = [float(m.group(1)) for m in re.finditer(pattern, text)]
    return vals[-1] if vals else None


def last_int(pattern, text):
    vals = [int(m.group(1).replace(",", "")) for m in re.finditer(pattern, text)]
    return vals[-1] if vals else None


def main():
    if len(sys.argv) not in (2, 3):
        print("usage: parse_train_log.py LOG [OUT_JSON]", file=sys.stderr)
        return 2
    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    text = log_path.read_text(encoding="utf-8", errors="replace")

    train_steps = None
    train_time_min = None
    tok_per_sec = None
    for m in re.finditer(
        r"(?m)^(\d+)/(\d+) train_loss: [0-9.]+ train_time: ([0-9.]+)m tok/s: ([0-9]+)",
        text,
    ):
        train_steps = int(m.group(1))
        train_time_min = float(m.group(3))
        tok_per_sec = int(m.group(4))

    stop_step = None
    stop_total_steps = None
    stop_train_time_s = None
    for m in re.finditer(
        r"stopping_early: wallclock_cap train_time: ([0-9.]+)ms step: (\d+)/(\d+)",
        text,
    ):
        stop_train_time_s = float(m.group(1)) / 1000.0
        stop_step = int(m.group(2))
        stop_total_steps = int(m.group(3))

    result = {
        "log": str(log_path),
        "model_params": last_int(r"model_params:(\d+)", text),
        "quat_mlp": (re.findall(r"quat_mlp:(True|False)", text) or [None])[-1],
        "quat_mlp_impl": (re.findall(r"quat_mlp_impl:([A-Za-z0-9_.-]+)", text) or [None])[-1],
        "train_shards": last_int(r"train_shards: (\d+)", text),
        "val_tokens": last_int(r"val_tokens: (\d+)", text),
        "train_steps_last": train_steps,
        "train_time_min_last": train_time_min,
        "tok_per_sec_last": tok_per_sec,
        "wallclock_stop_step": stop_step,
        "wallclock_stop_total_steps": stop_total_steps,
        "wallclock_stop_train_time_s": stop_train_time_s,
        "train_steps_final": stop_step if stop_step is not None else train_steps,
        "prequant_val_bpb": last_float(
            r"diagnostic pre-quantization post-ema val_loss:[0-9.]+ val_bpb:([0-9.]+)",
            text,
        ),
        "quantized_val_bpb": last_float(
            r"diagnostic quantized val_loss:[0-9.]+ val_bpb:([0-9.]+)",
            text,
        ),
        "quantized_ttt_val_bpb": last_float(
            r"quantized_ttt_phased val_loss:[0-9.]+ val_bpb:([0-9.]+)",
            text,
        ),
        "quantized_model_bytes": last_int(
            r"Serialized model quantized\+[^:]+: ([0-9,]+) bytes", text
        ),
        "total_submission_bytes": last_int(
            r"Total submission size quantized\+[^:]+: ([0-9,]+) bytes", text
        ),
        "peak_memory_mib": last_int(r"peak memory allocated: ([0-9,]+) MiB", text),
        "total_eval_time_s": last_float(r"total_eval_time:([0-9.]+)s", text),
        "artifact_under_16mb": None,
        "status_hint": "ok",
    }
    if result["total_submission_bytes"] is not None:
        result["artifact_under_16mb"] = result["total_submission_bytes"] < 16_000_000
    if "Traceback (most recent call last):" in text or "RuntimeError:" in text:
        result["status_hint"] = "error"

    payload = json.dumps(result, indent=2, sort_keys=True)
    if out_path is not None:
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
