#!/usr/bin/env python3
"""Extract stable metrics from a lean A40 harness run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable


PATTERNS = {
    "model_params": re.compile(r"^model_params:(?P<value>\d+)\s*$"),
    "prequant": re.compile(
        r"^diagnostic pre-quantization post-ema "
        r"val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+) "
        r"eval_time:(?P<eval_ms>[0-9.]+)ms"
    ),
    "quantized": re.compile(
        r"^diagnostic quantized "
        r"val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+) "
        r"eval_time:(?P<eval_ms>[0-9.]+)ms"
    ),
    "serialized_quant": re.compile(
        r"^Serialized model quantized\+(?P<compressor>[^:]+): "
        r"(?P<bytes>\d+) bytes"
    ),
    "total_submission": re.compile(
        r"^Total submission size quantized\+(?P<compressor>[^:]+): "
        r"(?P<bytes>\d+) bytes"
    ),
    "peak_memory": re.compile(
        r"^peak memory allocated: (?P<allocated>\d+) MiB "
        r"reserved: (?P<reserved>\d+) MiB"
    ),
    "stop": re.compile(
        r"^stopping_early: wallclock_cap train_time: "
        r"(?P<train_ms>[0-9.]+)ms step: (?P<step>\d+)/(?P<iterations>\d+)"
    ),
    "train_loss": re.compile(
        r"^(?P<step>\d+)/(?P<iterations>\d+) train_loss: .* "
        r"train_time: (?P<train_min>[0-9.]+)m tok/s: (?P<tok_s>[0-9.]+)"
    ),
    "ttt": re.compile(r"^(ttt_|ttp:|ttpr:|quantized_ttt_phased|total_eval_time:)"),
}


def read_key_values(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith("["):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


def iter_lines(path: Path) -> Iterable[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse(run_dir: Path) -> dict[str, str]:
    manifest = read_key_values(run_dir / "manifest.txt")
    command = read_key_values(run_dir / "command.txt")
    metrics: dict[str, str] = {}

    for key in (
        "job_id",
        "job_name",
        "host",
        "partition",
        "nodes",
        "cuda_visible_devices",
        "repo_path",
        "parameter_golf_path",
        "record_path",
        "vocab_size",
        "model_variant",
        "data_path",
        "tokenizer_path",
        "artifact_dir",
        "run_id",
        "seed_value",
        "train_shards",
        "val_shards",
        "val_byte_shards",
        "git_commit",
        "parameter_golf_commit",
    ):
        if key in manifest:
            metrics[key] = manifest[key]

    for key in (
        "HARNESS_MODE",
        "MODEL_VARIANT",
        "VOCAB_SIZE",
        "QUAT_MLP",
        "QUAT_MLP_IMPL",
        "TTT_ENABLED",
        "DOCUMENT_PACKING",
        "TORCH_COMPILE",
        "FUSED_MLP_ENABLED",
        "FUSED_CE_ENABLED",
        "TRAIN_BATCH_TOKENS",
        "VAL_BATCH_TOKENS",
        "MAX_WALLCLOCK_SECONDS",
        "GPTQ_CALIBRATION_BATCHES",
        "COMPRESSOR",
    ):
        if key in command:
            metrics[key.lower()] = command[key]

    ttt_seen = False
    last_train_step = ""
    last_train_iterations = ""
    last_train_min = ""
    last_tok_s = ""

    for line in iter_lines(run_dir / "train.log"):
        if match := PATTERNS["model_params"].match(line):
            metrics["model_params"] = match.group("value")
        if match := PATTERNS["prequant"].match(line):
            metrics["prequant_val_loss"] = match.group("loss")
            metrics["prequant_val_bpb"] = match.group("bpb")
            metrics["prequant_eval_ms"] = match.group("eval_ms")
        if match := PATTERNS["quantized"].match(line):
            metrics["quantized_val_loss"] = match.group("loss")
            metrics["quantized_val_bpb"] = match.group("bpb")
            metrics["quantized_eval_ms"] = match.group("eval_ms")
        if match := PATTERNS["serialized_quant"].match(line):
            metrics["compressor"] = match.group("compressor")
            metrics["quantized_model_bytes"] = match.group("bytes")
        if match := PATTERNS["total_submission"].match(line):
            metrics["compressor"] = match.group("compressor")
            metrics["total_submission_bytes"] = match.group("bytes")
        if match := PATTERNS["peak_memory"].match(line):
            metrics["peak_memory_allocated_mib"] = match.group("allocated")
            metrics["peak_memory_reserved_mib"] = match.group("reserved")
        if match := PATTERNS["stop"].match(line):
            metrics["train_time_ms"] = match.group("train_ms")
            metrics["train_steps"] = match.group("step")
            metrics["train_iterations"] = match.group("iterations")
        if match := PATTERNS["train_loss"].match(line):
            last_train_step = match.group("step")
            last_train_iterations = match.group("iterations")
            last_train_min = match.group("train_min")
            last_tok_s = match.group("tok_s")
        if PATTERNS["ttt"].match(line):
            ttt_seen = True

    if "train_steps" not in metrics and last_train_step:
        metrics["train_steps"] = last_train_step
        metrics["train_iterations"] = last_train_iterations
        metrics["last_train_time_min"] = last_train_min
        metrics["last_tok_s"] = last_tok_s

    metrics["ttt_seen"] = "1" if ttt_seen else "0"
    metrics["complete_file"] = "1" if (run_dir / "COMPLETE.txt").exists() else "0"
    metrics["train_log_exists"] = "1" if (run_dir / "train.log").exists() else "0"
    return metrics


def write_outputs(metrics: dict[str, str], run_dir: Path) -> None:
    keys = sorted(metrics)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.env").write_text(
        "".join(f"{key}={metrics[key]}\n" for key in keys),
        encoding="utf-8",
    )
    (run_dir / "metrics.tsv").write_text(
        "\t".join(keys) + "\n" + "\t".join(metrics[key] for key in keys) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics = parse(run_dir)
    write_outputs(metrics, run_dir)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()

