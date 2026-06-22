#!/usr/bin/env python3
"""Write minimal deterministic uint16 shards for package-size smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
BOS_ID = 1


def _write_shard(out_path: Path, arr: np.ndarray) -> None:
    if arr.dtype != np.uint16:
        raise TypeError(f"{out_path}: expected uint16, got {arr.dtype}")
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(arr.size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(arr.tobytes())


def _tokens(n: int, vocab_size: int, stride: int) -> np.ndarray:
    if vocab_size <= 8:
        raise ValueError(f"vocab_size must be >8, got {vocab_size}")
    usable = vocab_size - 4
    arr = ((np.arange(n, dtype=np.uint32) * 131 + 17) % usable + 4).astype(np.uint16)
    arr[::stride] = BOS_ID
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", required=True, type=Path)
    ap.add_argument("--vocab-size", required=True, type=int)
    ap.add_argument("--train-tokens", type=int, default=262_144)
    ap.add_argument("--val-tokens", type=int, default=65_536)
    args = ap.parse_args()

    train = _tokens(args.train_tokens, args.vocab_size, stride=257)
    val = _tokens(args.val_tokens, args.vocab_size, stride=257)
    val_bytes = np.ones(args.val_tokens, dtype=np.uint16)
    val_bytes[::257] = 0

    _write_shard(args.data_path / "fineweb_train_000000.bin", train)
    _write_shard(args.data_path / "fineweb_val_000000.bin", val)
    _write_shard(args.data_path / "fineweb_val_bytes_000000.bin", val_bytes)
    print(
        "wrote dummy shards "
        f"data_path={args.data_path} train_tokens={train.size} val_tokens={val.size}"
    )


if __name__ == "__main__":
    main()
