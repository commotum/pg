#!/usr/bin/env python3
"""Summarize goal-2 Phase 4 benchmark matrix outputs."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_RUN_ROOT = Path("/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks")
DEFAULT_CAP = 16000000


def read_env(path):
    data = {}
    for line in path.read_text().splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def sample_std(values):
    if len(values) < 2:
        return None
    mu = mean(values)
    assert mu is not None
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def fmt(value, digits=8):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def read_ledger(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            vocab, model, seed, job_id, submitted_at, smoke_metrics = parts[:6]
            rows[job_id] = {
                "vocab_size": vocab,
                "model_variant": model,
                "seed_value": seed,
                "job_id": job_id,
                "submitted_at": submitted_at,
                "smoke_metrics": smoke_metrics,
            }
    return rows


def collect_runs(run_root, cap):
    ledger = read_ledger(run_root / "submitted.tsv")
    runs = {job_id: dict(row) for job_id, row in ledger.items()}

    for metrics_path in sorted(run_root.glob("*/metrics.env")):
        job_id = metrics_path.parent.name
        metrics = read_env(metrics_path)
        row = runs.setdefault(job_id, {"job_id": job_id})
        row.update(metrics)
        row["metrics_path"] = str(metrics_path)
        row["run_dir"] = str(metrics_path.parent)

    for job_id, row in runs.items():
        row.setdefault("job_id", job_id)
        row.setdefault("metrics_path", "")
        row.setdefault("run_dir", str(run_root / job_id))
        total_bytes = as_int(row.get("total_submission_bytes"))
        complete = row.get("complete_file") == "1"
        ttt_ok = row.get("ttt_seen") == "0"
        has_bpb = as_float(row.get("quantized_val_bpb")) is not None
        row["compliant_under_cap"] = "1" if total_bytes is not None and total_bytes <= cap else "0"
        row["benchmark_complete"] = "1" if complete and ttt_ok and has_bpb else "0"
        if not row["metrics_path"]:
            status = "missing_metrics"
        elif not complete:
            status = "incomplete"
        elif not ttt_ok:
            status = "ttt_seen"
        elif total_bytes is None:
            status = "missing_size"
        elif total_bytes > cap:
            status = "over_budget"
        elif not has_bpb:
            status = "missing_bpb"
        else:
            status = "complete"
        row["qa_status"] = status
    return sorted(
        runs.values(),
        key=lambda row: (
            as_int(row.get("vocab_size")) or 0,
            row.get("model_variant", ""),
            as_int(row.get("seed_value")) if as_int(row.get("seed_value")) is not None else 999999,
            row.get("job_id", ""),
        ),
    )


def summarize(runs):
    groups = defaultdict(list)
    for row in runs:
        if row.get("benchmark_complete") == "1":
            groups[(row.get("vocab_size", ""), row.get("model_variant", ""))].append(row)

    out = []
    for (vocab, model), rows in sorted(groups.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        quant = [as_float(row.get("quantized_val_bpb")) for row in rows]
        prequant = [as_float(row.get("prequant_val_bpb")) for row in rows]
        steps = [as_float(row.get("train_steps")) for row in rows]
        quant_values = [value for value in quant if value is not None]
        prequant_values = [value for value in prequant if value is not None]
        step_values = [value for value in steps if value is not None]
        over_budget = sum(1 for row in rows if row.get("compliant_under_cap") != "1")
        seeds = ",".join(sorted(row.get("seed_value", "") for row in rows))
        out.append(
            {
                "vocab_size": vocab,
                "model_variant": model,
                "completed_seeds": str(len(rows)),
                "seeds": seeds,
                "mean_quantized_val_bpb": fmt(mean(quant_values)),
                "std_quantized_val_bpb": fmt(sample_std(quant_values)),
                "mean_prequant_val_bpb": fmt(mean(prequant_values)),
                "std_prequant_val_bpb": fmt(sample_std(prequant_values)),
                "mean_train_steps": fmt(mean(step_values), digits=2),
                "over_budget_completed_runs": str(over_budget),
            }
        )
    return out


def paired_deltas(runs):
    by_key = {}
    for row in runs:
        if row.get("benchmark_complete") != "1":
            continue
        key = (row.get("vocab_size", ""), row.get("model_variant", ""), row.get("seed_value", ""))
        by_key[key] = row

    out = []
    vocabs = sorted({row.get("vocab_size", "") for row in runs if row.get("vocab_size")}, key=int)
    seeds = sorted({row.get("seed_value", "") for row in runs if row.get("seed_value")}, key=lambda s: int(s))
    for vocab in vocabs:
        for seed in seeds:
            dense = by_key.get((vocab, "dense", seed))
            qmlp = by_key.get((vocab, "qmlp", seed))
            if not dense or not qmlp:
                continue
            dense_bpb = as_float(dense.get("quantized_val_bpb"))
            qmlp_bpb = as_float(qmlp.get("quantized_val_bpb"))
            if dense_bpb is None or qmlp_bpb is None:
                continue
            out.append(
                {
                    "vocab_size": vocab,
                    "seed": seed,
                    "dense_job_id": dense.get("job_id", ""),
                    "qmlp_job_id": qmlp.get("job_id", ""),
                    "dense_quantized_val_bpb": fmt(dense_bpb),
                    "qmlp_quantized_val_bpb": fmt(qmlp_bpb),
                    "qmlp_minus_dense_bpb": fmt(qmlp_bpb - dense_bpb),
                    "dense_under_cap": dense.get("compliant_under_cap", ""),
                    "qmlp_under_cap": qmlp.get("compliant_under_cap", ""),
                }
            )
    return out


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, fields):
    if not rows:
        return "_No rows yet._\n"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path, runs, summaries, deltas):
    summary_fields = [
        "vocab_size",
        "model_variant",
        "completed_seeds",
        "seeds",
        "mean_quantized_val_bpb",
        "std_quantized_val_bpb",
        "mean_train_steps",
        "over_budget_completed_runs",
    ]
    run_fields = [
        "vocab_size",
        "model_variant",
        "seed_value",
        "job_id",
        "qa_status",
        "quantized_val_bpb",
        "prequant_val_bpb",
        "train_steps",
        "total_submission_bytes",
        "host",
    ]
    delta_fields = [
        "vocab_size",
        "seed",
        "dense_quantized_val_bpb",
        "qmlp_quantized_val_bpb",
        "qmlp_minus_dense_bpb",
        "dense_under_cap",
        "qmlp_under_cap",
    ]
    text = [
        "# Goal 2 Matrix Summary",
        "",
        "Generated from Phase 4 benchmark metrics.",
        "",
        "## Cell Summaries",
        "",
        markdown_table(summaries, summary_fields),
        "## Paired Deltas",
        "",
        markdown_table(deltas, delta_fields),
        "## Run Rows",
        "",
        markdown_table(runs, run_fields),
    ]
    path.write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--byte-cap", type=int, default=DEFAULT_CAP)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.run_root / "matrix-summary")
    runs = collect_runs(args.run_root, args.byte_cap)
    summaries = summarize(runs)
    deltas = paired_deltas(runs)

    run_fields = sorted({key for row in runs for key in row})
    summary_fields = [
        "vocab_size",
        "model_variant",
        "completed_seeds",
        "seeds",
        "mean_quantized_val_bpb",
        "std_quantized_val_bpb",
        "mean_prequant_val_bpb",
        "std_prequant_val_bpb",
        "mean_train_steps",
        "over_budget_completed_runs",
    ]
    delta_fields = [
        "vocab_size",
        "seed",
        "dense_job_id",
        "qmlp_job_id",
        "dense_quantized_val_bpb",
        "qmlp_quantized_val_bpb",
        "qmlp_minus_dense_bpb",
        "dense_under_cap",
        "qmlp_under_cap",
    ]

    write_tsv(output_dir / "matrix-runs.tsv", runs, run_fields)
    write_tsv(output_dir / "matrix-summary.tsv", summaries, summary_fields)
    write_tsv(output_dir / "paired-deltas.tsv", deltas, delta_fields)
    write_markdown(output_dir / "matrix-summary.md", runs, summaries, deltas)
    print(output_dir / "matrix-summary.md")


if __name__ == "__main__":
    main()
