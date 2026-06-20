# Lightweight and Efficient Neural Natural Language Processing with Quaternion Networks

Source: arXiv 1906.04393, `https://arxiv.org/abs/1906.04393`

Authors: Yi Tay, Aston Zhang, Luu Anh Tuan, Jinfeng Rao, Shuai Zhang, Shuohang Wang, Jie Fu, Siu Cheung Hui

## Main Idea

The paper proposes replacing ordinary real-valued neural network projections with quaternion-valued projections. A quaternion has four real components, usually written as:

```text
Q = r + x i + y j + z k
```

The central claim is that Hamilton-product structure gives useful cross-component interactions while reducing learned degrees of freedom. A dense real matrix connecting four input components to four output components has 16 independent blocks. A quaternion matrix uses only four learned blocks and reuses them with fixed signs and permutations, giving roughly a 75% parameter reduction for compatible layers.

The authors apply this idea to two NLP building blocks:

- Quaternion feed-forward layers.
- Quaternion attention and Quaternion Transformer variants.

They argue that quaternion components act like structured multi-view or multi-head representations, but unlike independent heads, the Hamilton product forces interactions between the four components.

## Method

For a quaternion weight `W = W_r + W_x i + W_y j + W_z k` and quaternion input `Q = r + x i + y j + z k`, the Hamilton product can be written as a structured block matrix:

```text
[ W_r  -W_x  -W_y  -W_z ] [ r ]
[ W_x   W_r  -W_z   W_y ] [ x ]
[ W_y   W_z   W_r  -W_x ] [ y ]
[ W_z  -W_y   W_x   W_r ] [ z ]
```

Only `W_r`, `W_x`, `W_y`, and `W_z` are learned. The rest of the effective matrix is determined by the Hamilton-product algebra.

The paper uses this in three places:

- Quaternion FFN: replace dense feed-forward projections with Hamilton-product projections, then apply component-wise nonlinearities.
- Quaternion attention for pairwise sentence tasks: compute alignment with Hamilton products, apply component-wise softmax, and compare using quaternion feed-forward layers.
- Quaternion Transformer: replace Q/K/V projections and optionally the Transformer FFN projections with quaternion projections. The paper calls the Q/K/V-only version "partial" and the all-linear-projection version "full".

Real-valued inputs are treated as concatenated quaternion components `[r; x; y; z]`. Outputs can also be concatenated back into real space for ordinary softmax or sequence losses.

For initialization, the paper mentions specialized quaternion initialization but reports that independent Glorot initialization for the four components worked as well or better in their NLP experiments.

## Key Results

The headline result is that quaternion models often match or beat similarly shaped real-valued baselines while using fewer parameters.

Pairwise text classification and ranking:

- Quaternion attention at `d=50` is compared against Decomposable Attention.
- It is usually close to the larger DeAtt `d=200` baseline, with about 68-71% fewer parameters.
- It substantially beats a parameter-matched smaller DeAtt baseline in the authors' description.

Sentiment analysis:

- Transformer: 400K parameters, IMDb 82.6, SST 78.9.
- Quaternion Transformer full: 100K parameters, IMDb 83.9, SST 80.5.
- Quaternion Transformer partial: 300K parameters, IMDb 83.6, SST 81.4.

Machine translation:

- Transformer Base: 44M non-embedding parameters.
- Quaternion full: 11M parameters, typically close but can lose several BLEU points.
- Quaternion partial: 29M parameters, close or better on the reported En-Vi and En-Et tasks, nearly tied on En-Ro.
- The authors also report WMT14 En-De BLEU of 26.42 for partial and 25.14 for full in single-GPU training.

Mathematical language understanding:

- Transformer: 76.1 sequence accuracy with 400K parameters.
- Quaternion full: 78.9 with 100K parameters.
- Quaternion partial: 84.4 with 300K parameters.

Subject-verb agreement:

- Transformer: 94.8 accuracy with 400K parameters.
- Quaternion full: 94.7 with 100K parameters.
- Quaternion partial: 95.5 with 300K parameters.

## Assumptions and Limitations

The experiments are not modern large-scale language-model pretraining experiments. They cover smaller Tensor2Tensor Transformers, pairwise classification, translation, mathematical transduction, and agreement classification.

The reported parameter savings are raw learned-parameter savings. They do not directly measure wall-clock speed, kernel efficiency, compressed artifact size, or quality under a strict training-time cap.

The full quaternion self-attention variant changes attention computation itself with component-wise softmax over quaternion attention scores. That is less likely to use modern fused attention kernels directly and may be a poor fit for speed-constrained GPU training.

The four-component grouping is an architectural prior. It assumes the model benefits from fixed cross-component sharing. If the task needs more flexible independent channel mixing, the constraint can underfit.

Some reported results are mixed: full quaternion Transformers lose more clearly on WMT En-Ro, while partial quaternionization often looks like the safer tradeoff.

## Implementation-Relevant Details

The practical core is a `QuaternionLinear` layer whose input and output dimensions are divisible by 4. If input dimension is `4 * in_q` and output dimension is `4 * out_q`, the layer stores four matrices of shape `[out_q, in_q]` instead of one matrix of shape `[4 * out_q, 4 * in_q]`.

For chunks `(r, x, y, z)`, the output chunks are:

```text
o_r = W_r r - W_x x - W_y y - W_z z
o_x = W_x r + W_r x - W_z y + W_y z
o_y = W_y r + W_z x + W_r y - W_x z
o_z = W_z r - W_y x + W_x y + W_r z
```

A straightforward implementation using many small matmuls may be slower despite fewer parameters. A good implementation should benchmark:

- Four learned component matrices as separate 2D parameters.
- A fused or batched matmul formulation where possible.
- Whether `torch.compile` can fuse the chunk/sign/add operations.
- Whether reduced parameter count offsets any extra matmul overhead.

The paper's safest variant for modern causal LM training is probably not full quaternion attention. A lower-risk first experiment is to replace selected dense projections while keeping standard scaled dot-product attention:

- Q/K/V projections only.
- MLP `fc` and `proj` only.
- Q/K/V plus MLP projections.

## Relevance to This Repo

This repo contains the Parameter Golf submodule. The active baseline in `parameter-golf/train_gpt.py` is a compact causal Transformer with:

- `MODEL_DIM=512`.
- `NUM_HEADS=8`.
- `NUM_KV_HEADS=4`.
- `MLP_MULT=2`.
- `CastedLinear` projections for Q/K/V, output projection, and MLP projections.
- Tied token embeddings by default.

Those dimensions are compatible with quaternion grouping:

- `512 / 4 = 128`.
- KV projection size is `4 * 64 = 256`, also divisible by 4.
- MLP hidden size is `2 * 512 = 1024`, also divisible by 4.

A Parameter Golf experiment could add a `QuaternionLinear` replacement for `CastedLinear` and gate it by environment variables, for example:

- `QUAT_ATTN=1` to replace `c_q`, `c_k`, `c_v`, and maybe `proj`.
- `QUAT_MLP=1` to replace `MLP.fc` and `MLP.proj`.

Implementation trap: `train_gpt.py` sends only 2D block parameters to Muon:

```text
p.ndim == 2 and not control tensor
```

If quaternion weights are stored as a single 3D tensor like `[4, out_q, in_q]`, they will not be picked up by the existing Muon optimizer split. Store the four component matrices as separate 2D `nn.Parameter`s or update the optimizer grouping intentionally.

Another trap: the challenge metric is not raw parameter count. The final artifact is compressed and evaluated by bits per byte. Quaternion sharing is attractive because it reduces stored weights, but it still needs to survive:

- training speed limits,
- compressed artifact accounting,
- quantization/export code paths,
- and actual validation BPB.

The most pragmatic first test would be `QUAT_MLP=1` with standard attention unchanged. It directly targets large dense matrices, preserves fused SDPA, and avoids changing attention semantics. If that is stable, try Q/K/V projection replacement next. Full quaternion component-wise attention should be treated as a separate research branch because it is more likely to disrupt speed.

