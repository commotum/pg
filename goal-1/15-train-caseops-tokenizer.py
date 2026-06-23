#!/usr/bin/env python3
"""Train a CaseOps SentencePiece tokenizer for a candidate vocab size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterator


def _iter_docs(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["text"] if isinstance(obj, dict) else obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", required=True, type=Path)
    ap.add_argument("--record-path", required=True, type=Path)
    ap.add_argument("--out-model", required=True, type=Path)
    ap.add_argument("--vocab-size", required=True, type=int)
    ap.add_argument("--tokenizer-train-docs", type=int, default=500_000)
    args = ap.parse_args()

    if args.out_model.suffix != ".model":
        raise ValueError(f"--out-model must end in .model, got {args.out_model}")
    if args.vocab_size <= 0 or args.vocab_size > 65536:
        raise ValueError(f"unsupported vocab size for uint16 shards: {args.vocab_size}")

    sys.path.insert(0, str(args.record_path.resolve()))
    from lossless_caps import (  # noqa: PLC0415
        LOSSLESS_CAPS_CASEOPS_V1,
        encode_lossless_caps_v2,
        get_text_transform_control_symbols,
    )
    import sentencepiece as spm  # noqa: PLC0415

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    out_vocab = args.out_model.with_suffix(".vocab")
    for artifact in (args.out_model, out_vocab):
        if artifact.exists():
            artifact.unlink()

    control_symbols = get_text_transform_control_symbols(LOSSLESS_CAPS_CASEOPS_V1)
    max_docs = args.tokenizer_train_docs if args.tokenizer_train_docs > 0 else None

    def sentences() -> Iterator[str]:
        for i, text in enumerate(_iter_docs(args.docs)):
            if max_docs is not None and i >= max_docs:
                break
            yield encode_lossless_caps_v2(text)

    prefix = args.out_model.with_suffix("")
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences(),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=0.999,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="nmt_nfkc",
        add_dummy_prefix=False,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=control_symbols,
        hard_vocab_limit=False,
    )

    sp = spm.SentencePieceProcessor(model_file=str(args.out_model))
    actual_vocab = int(sp.vocab_size())
    if actual_vocab != args.vocab_size:
        raise RuntimeError(
            f"trained tokenizer vocab_size={actual_vocab}, expected {args.vocab_size}"
        )

    manifest = {
        "vocab_size": actual_vocab,
        "model_path": str(args.out_model),
        "vocab_path": str(out_vocab),
        "tokenizer_train_docs": args.tokenizer_train_docs,
        "text_transform": LOSSLESS_CAPS_CASEOPS_V1,
        "control_symbols": control_symbols,
    }
    args.out_model.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
