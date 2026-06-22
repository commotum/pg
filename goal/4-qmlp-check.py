from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def load_train_gpt():
    path = Path("train_gpt.py").resolve()
    spec = importlib.util.spec_from_file_location("train_gpt_qmlp_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_params(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def main() -> None:
    tg = load_train_gpt()

    q1 = tg.QuaternionLinear(512, 1024)
    x1 = torch.randn(2, 3, 512)
    y1 = q1(x1)
    assert y1.shape == (2, 3, 1024), y1.shape
    y1.square().mean().backward()
    for name in ("wr", "wi", "wj", "wk"):
        grad = getattr(q1, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name

    q2 = tg.QuaternionLinear(1024, 512)
    x2 = torch.randn(2, 1024)
    y2 = q2(x2)
    assert y2.shape == (2, 512), y2.shape

    qsmall = tg.QuaternionLinear(8, 12, impl="split")
    qsmall_matrix = tg.QuaternionLinear(8, 12, impl="matrix")
    qsmall_matrix.load_state_dict(qsmall.state_dict())
    xsmall = torch.randn(5, 8)
    wr, wi, wj, wk = qsmall.wr, qsmall.wi, qsmall.wj, qsmall.wk
    full = torch.cat(
        (
            torch.cat((wr, -wi, -wj, -wk), dim=1),
            torch.cat((wi, wr, -wk, wj), dim=1),
            torch.cat((wj, wk, wr, -wi), dim=1),
            torch.cat((wk, -wj, wi, wr), dim=1),
        ),
        dim=0,
    )
    split_err = (qsmall(xsmall) - torch.nn.functional.linear(xsmall, full)).abs().max().item()
    matrix_err = (qsmall_matrix(xsmall) - qsmall(xsmall)).abs().max().item()
    assert split_err < 1e-6, split_err
    assert matrix_err < 1e-6, matrix_err

    dense = tg.GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        quat_mlp=False,
        quat_mlp_impl="split",
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    qmodel = tg.GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        quat_mlp=True,
        quat_mlp_impl="matrix",
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    dense_params = count_params(dense)
    q_params = count_params(qmodel)
    saved = dense_params - q_params
    assert dense_params == 17_059_912, dense_params
    assert saved == 7_077_888, saved
    assert q_params == 9_982_024, q_params

    matrix_param_names = {
        name
        for name, param in qmodel.blocks.named_parameters()
        if param.ndim == 2 and not any(pattern in name for pattern in tg.CONTROL_TENSOR_NAME_PATTERNS)
    }
    qmlp_names = {
        name
        for name, _ in qmodel.blocks.named_parameters()
        if ".mlp." in name and name.rsplit(".", 1)[-1] in {"wr", "wi", "wj", "wk"}
    }
    missing = sorted(qmlp_names - matrix_param_names)
    assert not missing, missing

    toy = tg.GPT(
        vocab_size=32,
        num_layers=2,
        model_dim=16,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        quat_mlp=True,
        quat_mlp_impl="matrix",
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    input_ids = torch.randint(0, 32, (2, 8))
    target_ids = torch.randint(0, 32, (2, 8))
    loss = toy(input_ids, target_ids)
    assert torch.isfinite(loss), loss
    loss.backward()

    print(f"shape_checks=ok")
    print(f"hamilton_equivalence_max_error={split_err:.3g}")
    print(f"matrix_split_equivalence_max_error={matrix_err:.3g}")
    print(f"dense_params={dense_params}")
    print(f"qmlp_params={q_params}")
    print(f"saved_params={saved}")
    print(f"qmlp_muon_matrix_params={len(qmlp_names)}")
    print(f"toy_forward_backward=ok loss={float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()
