#!/usr/bin/env python3
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


def module_status(name):
    return importlib.util.find_spec(name) is not None


def tokenizer_vocab(path):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=path)
    return int(sp.vocab_size())


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = {
        "python": sys.version,
        "modules": {
            name: module_status(name)
            for name in [
                "torch",
                "triton",
                "sentencepiece",
                "brotli",
                "flash_attn_interface",
            ]
        },
        "lrzip": shutil.which("lrzip"),
        "tokenizers": {},
        "cuda": {},
    }

    required = [name for name, ok in result["modules"].items() if not ok]
    if required:
        result["status"] = "failed"
        result["error"] = f"missing modules: {', '.join(required)}"
    else:
        import torch
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

        result["flash_attn_func"] = str(flash_attn_func)
        result["flash_attn_varlen_func"] = str(flash_attn_varlen_func)
        result["torch_version"] = torch.__version__
        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
        expected_devices = int(os.environ.get("GOAL3_EXPECT_CUDA_DEVICES", "8"))
        if torch.cuda.device_count() != expected_devices:
            result["status"] = "failed"
            result["error"] = (
                f"expected {expected_devices} CUDA devices, "
                f"got {torch.cuda.device_count()}"
            )
        elif result["lrzip"] is None:
            result["status"] = "failed"
            result["error"] = "lrzip not found on PATH"
        else:
            checks = [
                ("sp8192", 8192, os.environ["GOAL3_SP8192_TOKENIZER"]),
                ("sp16384", 16384, os.environ["GOAL3_SP16384_TOKENIZER"]),
            ]
            for label, expected, path in checks:
                got = tokenizer_vocab(path)
                result["tokenizers"][label] = {
                    "path": path,
                    "expected_vocab_size": expected,
                    "vocab_size": got,
                }
                if got != expected:
                    result["status"] = "failed"
                    result["error"] = f"{label} expected vocab {expected}, got {got}"
                    break
            else:
                result["status"] = "passed"

    payload = json.dumps(result, indent=2, sort_keys=True)
    if out_path is not None:
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
