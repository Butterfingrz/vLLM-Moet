# Kimi-K2.7-Code (1T, NVFP4) on SM120

[nvidia/Kimi-K2.7-Code-NVFP4](https://huggingface.co/nvidia/Kimi-K2.7-Code-NVFP4)
(595 GB, modelopt NVFP4) served through the vLLM-Moet 2-bit expert stack on SM120 —
TP4 GPU-resident on 4× RTX PRO 6000, and the BASE-cache (host-resident) path for
2-card configs. The stock checkpoint cannot fit this box at all (595 GB weights vs
384 GB total VRAM), so as with GLM-5.2 the 2-bit path is the capacity unlock, not
just a speedup.

This documents the port: what the checkpoint is, what changed, the bring-up findings
(several real bugs, one upstream-relevant), and the measured results the release
tables import. The serve configs are the recipes in
`bench/recipes/kimi-k2.7-code-nvfp4/` — the recipes image runs them directly.

## Measured results (2026-07-10)

**TP4, 4× RTX PRO 6000 (96 GB), GPU-resident 2-bit experts + FP4 delta (auto ≈2.7 GiB/rank),
262144-token window (util 0.94, 292.6K-token fp8 KV pool), CUDA graphs FULL_AND_PIECEWISE:**

| probe | result |
|---|---|
| decode, single stream, 512 tok greedy | **51 tok/s** (median of 3; no MTP — checkpoint ships none) |
| batched decode (256 tok/stream, aggregate) | 1→**52**, 4→**149**, 8→**222 tok/s** (27.7/stream at 8) |
| prefill, 8K unique prompt | **2 448 tok/s** (median of 3; AFRAG on) |
| needle 8K / 32K / 80K / **128K** (depths 0.1-0.5) | **PASS** (128K: TTFT+gen 84.5 s cold) |
| needle post-stride-fix, default 64Ki workspace: 80K / 128K / **192K** / **248K** | **PASS** (2-4 context chunks exercised; 248K: TTFT+gen 629 s cold, reply drifts after the correct passphrase — quality at the YaRN edge, not a retrieval failure) |
| arithmetic (5× multi-digit, chat+reasoning) | 5/5 |
| code-gen (`top_k_frequent`, executed) | tests pass; clean single code block |
| tool calling (`kimi_k2` parser, auto tool choice) | call + round-trip **PASS** (`--enable-auto-tool-choice` required) |
| coherence | greedy outputs coherent across all probes |
| load time (595 GB checkpoint → planes) | ~25 min (4:40 shard read + ~9 min f64 requant/staging + init) |

**TP2 + BASE cache (2× 96 GB, 52 GiB/rank pool ≈ 39% coverage, 16K window):** serves
coherently at **14.4 tok/s** decode with pool still converging (miss replays); needle 8K
PASS. The TP2 path is functional but wants the planes cache + longer warm before real
numbers.

Open follow-ups: FP8 dense (below), upstreaming the `merge_attn_states` stride fix
(finding 4 below). Planes cache + Eagle3 drafter: shipped and serving.

## What the checkpoint is

| fact | value | consequence |
|---|---|---|
| architecture | `KimiK25ForConditionalGeneration` (VLM wrapper), text = `DeepseekV3ForCausalLM` | v0.24.0 already registers it (`model_executor/models/kimi_k25.py`); text model instantiated as classic `DeepseekV2ForCausalLM` → standard `FusedMoE` → `ModelOptNvFp4FusedMoE`. **Our existing modelopt hook is on exactly this class.** |
| geometry | 61 layers (layer 0 dense), H=7168, 384 routed experts top-8 + 1 shared, I=2048, MLA (q_lora 1536 / kv_lora 512 / rope 64), vocab 163 840 | **H=7168 needed a new cubin family** — everything else was in the shipped K set. |
| quantization | modelopt NVFP4 (e2m1 + e4m3 block-16 scales + per-tensor `weight_scale_2` + `input_scale`), **routed experts only** | same format as GLM-5.2-NVFP4 → `build_layer_planes_nvfp4` / `nvfp4_to_codes_scales` reuse as-is. |
| everything else | BF16 (attention, shared experts, dense layer 0, embed/lm_head, vision tower); `exclude_modules` lists them, `hf_quant_config.json` | excluded modules resolve to `UnquantizedLinearMethod` — no DeepGEMM/fp8-einsum involvement at all (unlike DS4). BF16 dense is ~3.1× the per-token bytes of the 2-bit experts → main perf lever later (FP8 follow-up below). |
| attention | **dense** MLA (no DSA/Lightning indexer), 256K YaRN | SM120 backend = `TRITON_MLA` (supports fp8 KV). The sparse-MLA fixes, cubit MLA-prefill and NVFP4-KV tier in this repo do **not** apply — they are sparse-layout-specific. |
| MTP | `num_nextn_predict_layers: 0`, no drafter weights | no MTP speedup available. Model declares `SupportsEagle3` — the external-drafter option is measured below. |
| tokenizer / serving | tiktoken custom code, `chat_template.jinja`, `kimi_k2` tool+reasoning parsers exist in v0.24.0 | serve with `--trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2`; **no** `--tokenizer-mode deepseek_v4`, **no** MTP speculative-config. |
| vision | MoonViT 27 blocks BF16, images+video | supported by upstream `kimi_k25_vit.py`; not yet validated on SM120 — text-only serving is the primary Code use case (`--limit-mm-per-prompt '{"image":0,"video":0}'` if it misbehaves). |

Checkpoint naming (`language_model.model.layers.N.mlp.experts.N.{gate,up,down}_proj.*`)
matches the vLLM module tree of the wrapper (prefix `language_model.`), and the
`exclude_modules` wildcards are in the same namespace — the modelopt exclusion matching
and `apply_vllm_mapper` expansion work unchanged. `mlp.gate` (router) is not excluded and
not quantized: `GateLinear` never asks for a quant method, so nothing to do.

## Memory & bandwidth budget (computed, not measured)

2-bit planes: 12.39 MB/expert-layer (codes13 7.34 + sc13 0.92 + codes2 3.67 + sc2 0.46)
× 384 experts × 60 layers = **265.8 GiB** total (vs 531.6 GiB raw NVFP4).

| config | planes/rank | BF16 dense/rank | total/rank | verdict |
|---|---:|---:|---:|---|
| TP4 (4× 96 GB) | 66.4 GiB | ~8 GiB | ~75 GiB | **fits GPU-resident**, ~15-18 GiB left for KV + FP4 delta + graphs |
| TP2 (2× 96 GB) | 132.9 GiB | ~13 GiB | >96 GiB | BASE cache: pool ~40-45 GiB/rank ≈ 30-34% expert coverage |
| TP1 (1× 96 GB) | 265.8 GiB | ~24 GiB | ≫96 GiB | BASE cache: pool ~50-55 GiB ≈ 19-21% coverage (the DS4-on-5090 regime); needs ~266 GiB pinned host (box has 1.5 TB) |

KV (fp8, dense MLA, replicated per TP rank): 34.3 KiB/token → 128K ctx = 4.3 GiB,
256K = 8.6 GiB. TP4 can hold the full 256K window single-seq.

Decode ceiling at TP4 (bytes/token/rank ÷ ~1.6 TB/s): experts-2bit 1.49 GB + shared-BF16
1.32 GB + attn-BF16 3.08 GB + lm_head 0.59 GB ≈ 6.5 GB → **~250 tok/s theoretical, expect
~100-140 single-stream** (no MTP). Batch scaling mirrors DS4/GLM (experts amortize).

## What the port changed

Two patch-side edits plus one kernel family; everything else in the hook chain was
already shape-generic (`is_w2_layer` regex matches the
`language_model.model.layers.N.mlp.experts` prefix; `create_weights` host-staging,
`build_layer_planes_nvfp4` with per-expert `weight_scale_2` [E,2]/[E] handling,
`_workspaces` sized from `st["K13"]`/`st["K2"]`, delta/base tiers, gate, deterministic
unpermute — all unchanged; shared experts stay a separate BF16 module on CUDA).

1. **K=7168 cubin family** (the only real kernel work; gate/up contracts over K=H=7168
   at every TP degree, down-proj K=I/TP was already shipped). Generated with
   `gen/gen_moe_w2.py` / `gen/gen_moe_w4.py` (7168 = 56×128, NWARP=8 → KSLICE=896),
   assembled with cubit (public release; `-t tables/sm120.json`), all op-validated on
   RTX PRO 6000: mc1/mc2/mc4 + w4 worst_rel ≤ 3.0e-3 deterministic, AFRAG bit-identical
   to mc4. New tooling: `gen/culaunch.py` (standalone driver harness),
   `gen/moe_w2_afrag_check.py`.
2. **Loader probe list** — `moe_w2_cubit.py::_ensure_ready` probes K=7168;
   `_require_kernels` enforces presence at weight load (`_nwarp_for_k(7168)` already
   returned 8 — matches the generator).
3. **Layer cutoff via text config** — `_layer_cutoff()` unwraps composite VLM configs
   via `hf_config.get_text_config()` (KimiK25Config keeps `num_hidden_layers` on
   `.text_config`; the bare lookup silently fell back to 43, sending layers 43-60 down
   the stock path). Returns self for DS4/GLM — no behaviour change there.
   `VLLM_MOE_W2_NUM_LAYERS` remains as an override.
4. **Serving flags, not code**: drop MTP speculative-config and `--tokenizer-mode
   deepseek_v4`; add `kimi_k2` parsers. DeepGEMM/flashinfer pins are harmless but
   unnecessary for this model (no sparse MLA, no fp8 dense einsum).

## Bring-up findings

1. **Zero-sign balancing (checkpoint-specific, caught by the sweep gate):** the Kimi
   NVFP4 export writes **all exact zeros as +0** (13.3% of expert mass; GLM/DS4 are
   ±0-balanced) — the sign-preserving zero→±1 map would inject a **+0.134** unit-space
   bias/tensor, 3× the asymmetry bias that degenerates GLM. Fixed in the loader
   (`_f64_to_codes_scales`): one-signed zeros map ±1 alternating by k-parity
   (`VLLM_MOE_W2_ZERO_MODE={auto,sign,alt}`); L2 identical, net bias −0.000, balanced
   checkpoints bit-exact. Sweep (162 tensors, 9 layers × 6 experts × 3 projections):
   `{-4,-1,1,4}` wins 162/162, rel-RMS penalty vs per-tensor asym optimum +1.5%.
   Tool: `tools/sweep_nvfp4_codebook.py`.
2. **`no_slots` OOB at 384 experts (real bug, in-patch):** first TP4 run decoded fine
   (50 tok/s, coherent) but every prompt past ~96 tokens degenerated into token soup.
   Root cause: at prefill the tier-less desc path binds `ws["no_slots"]` — a **fixed
   256-row** `-1` table — as the slot row, while the desc kernel clamps expert ids to
   `n_experts-1` = **383** and reads past the tensor. OOB int32s ≥ 0 routed pairs to
   the FP4 tier, whose GEMM never launches at prefill → those rows kept stale
   workspace values. DS4 (256 experts) could never hit this; GLM (160) neither.
   Fix: size `no_slots` to the model's expert count. The glue test now runs the
   prefill tier explicitly (`T=160` PASS at H=7168/I=512).
3. **Triton MLA decode smem (SM12x, in-patch):** at DeepSeek dims (512+64 tile,
   num_stages=2) the kernel wants 100 KiB smem > SM120's 99 KiB limit →
   `num_stages=1` fallback on capability 12.x, same pattern as the existing
   `BLOCK_DMODEL>=1024` case.
4. **`merge_attn_states` stride bug (>64K chunked prefill, upstream-relevant,
   fixed in-patch):** needle passed to 65.6K prompt tokens and failed from ~68K — the
   boundary tracks the MLA **chunked-context workspace cap (64Ki tokens)**:
   final-chunk context ≤64Ki is single-chunk (PASS), above it
   `_compute_prefill_context` iterates ≥2 chunks and the output was corrupt. Root
   cause: **`merge_attn_states` (CUDA + Triton) indexes BOTH prefix and suffix with
   the single stride `prefix_output.stride(1)`**. On FA2 platforms (SM120 — no
   FA3/FA4) every chunk output is an unpadded slice of the v-padded buffer (head
   stride 192 at head size 128), while the merged intermediate from the previous
   merge is allocated contiguous via `empty_like` (stride 128). First merge diverges
   the strides → the final context⊕suffix merge (≥2 chunks) and the in-loop merges
   (≥3 chunks) read the suffix at wrong offsets → garbage. A faithful repro
   (per-chunk unpad slices, production loop) shows rel err 0.22 (2 chunks) / 0.72
   (3 chunks) vs 5e-3 single-chunk — and ~8e-3 after the fix. Fix in
   `vllm/v1/attention/ops/merge_attn_states.py`: contiguize prefix/suffix when their
   strides differ (no-op otherwise). Upstream master (Jul 2026) has the same bug —
   FA3/FA4 (Hopper+) never pad v, so strides always match there; affects FA2
   fallback platforms only. `VLLM_MLA_CHUNKED_WORKSPACE_TOKENS` stays as a diagnostic
   knob but is **no longer needed**: default 64Ki workspace + full 262144 window
   serve correctly — needle PASS at 80K/128K/192K/248K post-fix (2-4 context chunks),
   util back to 0.94.
5. **Box finding (not a code change):** GPU **P2P is silently broken** on this host
   (`can_device_access_peer`=True but D2D copies corrupt) → NCCL's first allreduce
   hangs at 100% util. **`NCCL_P2P_DISABLE=1` required** for any TP here (SHM
   transport; PIX topology, RunPod container).
6. **Venv-only (not in-patch):** Kimi's MLA rope goes through
   `DeepseekScalingRotaryEmbedding` → flashinfer JIT rope (rope head 64). On the
   no-docker venv, flashinfer 0.6.14 detects SM120 but the worker env lacked a CUDA
   ≥12.9 toolchain → `_normalize_cuda_arch` raised, `TARGET_CUDA_ARCHS` stayed empty,
   and the JIT died with the misleading "FlashInfer requires GPUs with sm75 or
   higher". Fix: `FLASHINFER_CUDA_ARCH_LIST=12.0f` + `CUDA_HOME`/`PATH`/
   `LD_LIBRARY_PATH` pointing at the pip `nvidia/cu13` toolkit (nvcc 13.2). The
   official docker image ships nvcc and never hits this.

## Serve config

The canonical configs are the recipes (`kimi-k2.7-code-nvfp4/pro6000x4-tp4-256k`,
`.../pro6000x4-tp4-eagle3`, `.../pro6000x2-basecache`). The validated no-docker venv
command (in the docker image the `FLASHINFER_*`/`CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH`
lines are unnecessary):

```bash
NCCL_P2P_DISABLE=1 \                       # this host: P2P silently corrupt
FLASHINFER_CUDA_ARCH_LIST="12.0f" \        # venv-only: rope JIT arch detect
CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13 \
PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_MOE_W2=1 VLLM_MOE_W2_DELTA_GB=auto VLLM_MOE_W2_DELTA_RESERVE_GB=6 \
VLLM_MOE_W2_CUBIT_DIR=/root/vLLM-Moet/kernels/cubins-sm120 \
VLLM_MOE_W2_PLANES_CACHE=/root/models/.planes-cache-kimi \  # optional: fast restarts
vllm serve /root/models/Kimi-K2.7-Code-NVFP4 \
  --served-model-name kimi-k2.7-code --trust-remote-code \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 262144 \
  --gpu-memory-utilization 0.94 --max-num-batched-tokens 2048 --max-num-seqs 8 \
  --no-scheduler-reserve-full-isl \
  --enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
```

(Full native 262144 window since the finding-4 fix: the default 64Ki chunked
workspace serves any context length; fp8 KV pool at util 0.94 = 292.6K tokens,
covers a full-window sequence. Chunked-context up-projection transients are
~0.55 GiB/rank and *outside* the memory profiler's view — with the small default
workspace, util 0.94 + `DELTA_RESERVE_GB=6` survives a cold 248K prefill.
`--enable-auto-tool-choice` is required for `"tool_choice": "auto"`.)

## BASE-cache configs (TP2 / TP1)

TP2 (~43-52 GiB/rank pool) and TP1×96GB (~53 GiB pool, ~266 GiB pinned host) run the
same three-tier stack as GLM (host 2-bit → GPU 2-bit cache → gate-filled FP4).
Routing concentration at **384 experts top-8 is a different regime** than DS4's 256 /
GLM's 160 — the 19%-coverage≈96%-hit heuristic must be re-established from the live
pool hit-rate counters before promising numbers. Current state: functional at
14.4 tok/s (see measured results), pool still converging.

## Eagle3 drafter (replaces the missing MTP)

The model declares `SupportsEagle3` and a matched community drafter exists —
[AQ-MedAI/Kimi-K2.7-Code-eagle3](https://huggingface.co/AQ-MedAI/Kimi-K2.7-Code-eagle3)
(2B, 1 layer, hidden 7168, target vocab 163840 ✓ vLLM's vocab-equality check,
draft_vocab 96000 + d2t/t2d, `LlamaForCausalLMEagle3` — registered in v0.24;
acceptance ~2.9 on HumanEval). Acceptance on code is excellent (mean acceptance
length 3.4-3.7 of 4, 82-90% draft acceptance — the generic-data drafter transfers
fine to the 2-bit target); the net depends on drafter sharding: with
`draft_tensor_parallel_size: 1` the unsharded 2B drafter (3 sequential passes/step,
96K-vocab head on one GPU) *lost* to baseline (~42 vs 51 tok/s; GPU0 42% util vs 95%
others); `draft_tensor_parallel_size: 4` wins. Keep k at 1 or 3 — MLA capture-size
issue at other values; drafter decodes run FULL_DECODE_ONLY graphs.

**Variance root cause: the FP4 delta tier makes greedy decode non-deterministic
across runs** — background promotions flip experts between 2-bit and FP4 mid-serving,
perturbing logits at near-ties; 6/6 identical greedy prompts produced 6 distinct
outputs (first divergence within ~100 chars). Without speculation this only wobbles
content, not throughput (baseline was ±1%); with speculation, content trajectory
drives acceptance → 30-60 tok/s spread. This is inherent to the delta tier's design
(runtime precision changes), not a bug — but it interacts badly with spec-decode
benchmarking and reproducibility. **A/B of spec variants therefore runs with
`VLLM_MOE_W2_DELTA_GB=0`.**

**Spec-decode A/B (2026-07-10, TP4, 256K window, delta off, greedy code prompt,
6 runs each):**

| variant | median decode | spread | deterministic | notes |
|---|---:|---:|---|---|
| no speculation | 51 tok/s | 1% | yes | |
| **eagle3 k=3 dtp=4** | **61.1 tok/s (+20%)** | 6% | **yes (6/6 identical)** | production pick |
| eagle3 k=2 dtp=4 | 57.4 tok/s | 3% | yes | verify nearly free → deeper drafts win |
| eagle3 k=3 dtp=1 | ~42 tok/s | high | — | unsharded 2B drafter starves one GPU |
| eagle3 k=3 + delta auto | 57.2 tok/s | high | no | delta costs ~4 tok/s + determinism |
| suffix k=32 / k=3, FULL graphs | fails to start | — | — | MLA `build_for_cudagraph_capture` asserts (`max_query_len <= reorder_batch_threshold`); vLLM v0.24 raises the threshold for eagle-family only |
| suffix k=16, PIECEWISE graphs | 56.6 tok/s | 17% | no (6/6 distinct) | AL 1.86, 35% accept — fresh code-gen has nothing to look up; variable-length verify shapes flip ties via reduction order; loses FULL-graph decode |

Suffix may still pay on real agent loops (re-emitted files/boilerplate) but is
structurally handicapped on this backend (no FULL graphs). Production config:
**eagle3 k=3, draft TP 4, delta off** (the `pro6000x4-tp4-eagle3` recipe). If
acceptance on real traffic disappoints (<2.3): train an Eagle3.1-MLA drafter with
TorchSpec/ModelOpt streaming against THIS NVFP4 deployment (~0.5-1k H200-hours,
CoreWeave's NVFP4-regenerated-data playbook).

## Follow-ups

- **FP8 dense + shared experts — ATTEMPTED, PARKED (2026-07-10)**: BF16 attention/
  shared/lm_head is ~77% of per-token bytes at TP4; FP8 would be ~+60% decode ceiling.
  The naive route (override `is_layer_excluded` → `Fp8LinearMethod` online quant,
  `VLLM_MOE_W2_DENSE_FP8={1,attn,l0}`, in-patch but default-off) is **blocked on two
  real issues** found in bring-up: (1) the stock non-serialized Fp8 flow depends on the
  loader's online-quantize hook that only engages when the TOP-LEVEL quantization is
  "fp8" — under the modelopt config `process_weights_after_loading` sees the sentinel
  scales and serves garbage (fixed in-patch with a self-quantizing subclass); (2) even
  then, the ScaledMM kernel wrapper **crashes with an illegal memory access on SM120**
  in an offline single-layer repro (CutlassFP8ScaledMM selected; the raw
  `ops.cutlass_scaled_mm` call with the same tensors computes fine, rel ~3e-2 — the
  fault is in the wrapper's padding/logical-size path, not the GEMM). Serving
  symptom: NaN logits → "!!!" floods in ALL scopes (attn/mlp/l0). Next steps:
  debug `ScaledMMLinearKernel.apply_weights` padding on SM120, or route through
  `skinny_fp8_cubit` (K∈{2048,4096,7168} repack list already covers Kimi), or
  torch._scaled_mm rowwise. Until then the flag stays off; production remains BF16
  dense.
- **Planes cache — SHIPPED + VALIDATED (v1)**: `VLLM_MOE_W2_PLANES_CACHE=<dir>` caches
  the built 2-bit planes (+FP4 delta planes when the tier is on) per TP rank
  (`moe_w2_planes_cache.py`; key = checkpoint sha + TP layout + zero mode + codebook
  version; per-layer hit/miss, async best-effort writes, size-validated reads). NVFP4
  builder only for now (mxfp4/fp8 builders: same two hooks). **Measured end-to-end
  (2026-07-10): write pass added no visible load time (739 GiB written async during
  the build); read pass hit 240/240 layers and cut the restart from ~25 min to
  ~14.5 min; needle + code output coherent on cache-loaded planes.** v1.5 follow-up:
  skip the expert shard read too (no-op `weight_loader` on validated hits) — would
  cut restarts to ~5 min.
- **Vision path validation** on SM120 (MoonViT BF16); until then serve text-only if it
  misbehaves (`--limit-mm-per-prompt '{"image":0,"video":0}'`).

## Explicitly out of scope

- MTP (checkpoint ships no drafter head — `num_nextn_predict_layers: 0`).
- NVFP4 KV cache tier (sparse-MLA `fp8_ds_mla`-layout-specific; Kimi is dense MLA).
- DeepGEMM / flashinfer sparse-MLA pins and the DS4 `o_proj`/indexer SM120 fixes (inert
  for this model; keep them in the shared image, they don't activate).
- EP / redundant experts / EPLB.
