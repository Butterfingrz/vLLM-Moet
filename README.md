# Frontier MoE on consumer Blackwell (SM120)

**Official vLLM v0.24.0 + a 5.6k‑line patch** that serves frontier Mixture‑of‑Experts models —
**GLM‑5.2 (753B)** and **DeepSeek‑V4‑Flash (159B)** — on consumer/workstation Blackwell
(RTX PRO 6000, RTX 5090), hardware their official checkpoints cannot even fit on. Three ideas
carry it:

1. **2‑bit experts with FP4 recovery** — routed experts compress to a sign‑symmetric 2‑bit
   codebook on hand‑written SM120 SASS kernels; a runtime FP4 tier (delta cache + confidence
   gate) restores precision exactly where it matters.
2. **Tiered expert residency** — when even the 2‑bit base outgrows VRAM, it moves to pinned
   host RAM and the GPU becomes an **expert cache** (miss → batched fetch + bit‑identical
   graph replay). That puts 753B on two 96 GB cards and 159B on a single RTX 5090.
3. **A rebuilt serving base** — vLLM v0.24.0 actually working on SM120 (the release is
   broken‑as‑shipped), plus MTP speculative decoding (incl. under pipeline parallelism,
   bit‑deterministic), an **NVFP4 KV cache** (352 B/token), and agent‑ready tool/reasoning
   parsing.

---

## GLM‑5.2 (753B) — the headline model

Served from the official [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4)
checkpoint (433 GB): the loader re‑quantizes modelopt NVFP4 experts (e2m1 × e4m3 block‑16 ×
per‑tensor scale_2) to the sign‑symmetric 2‑bit planes at load — f64‑exact vs the reference
pipeline on real shards. Single‑stream, greedy, CUDA graphs (2026‑07‑09/10):

| hardware | config | decode | context window |
|---|---|---:|---:|
| **4× RTX PRO 6000 (TP4)** | 2‑bit + **MTP k=2** | **105 tok/s** | 128K (331K fits) |
| 4× RTX PRO 6000 (TP4) | + FP4 delta + confidence gate | **83–85 tok/s** | 128K |
| **2× RTX PRO 6000 (TP2)** | **three‑tier residency** + MTP + gate | **45–52 tok/s** | 16K |

- **4 cards:** prefill ~2.5k tok/s; MTP acceptance 2.3–2.8; needle retrieval **PASS to 126K**
  on the nvfp4 KV cache and **to 276K** on fp8 (331K window fits at util 0.95). GLM's nominal
  1M window is KV‑bound on 4 cards. Tool calling (`glm47`) + reasoning (`glm45`) parsers work
  out of the box — the endpoint drives coding agents (opencode) directly.
- **2 cards — a model that doesn't fit, running anyway:** the 2‑bit planes alone (~190 GiB)
  match the entire 2‑GPU VRAM budget. **Three tiers** make it work: pinned‑host 2‑bit base →
  48 GiB/rank GPU expert cache (~51% coverage serving ~91% of token→expert routings warm) →
  a small gate‑filled FP4 pool for precision. With MTP k=2 (acceptance 3.0) and miss‑tolerance
  8: **45–52 tok/s** (strict zero‑tolerance: 21–24). Needle 4/4 PASS across the window.
  Bare‑2‑bit quality artifacts ("capital of Poland: Krakow", garbled Polish) are corrected by
  the FP4 tier. The window is KV‑budget‑bound, not architectural — the record config simply
  allotted KV 8 GiB/rank; a larger window trades against expert‑cache coverage.
- **NVFP4 KV cache** (`--kv-cache-dtype nvfp4`): packed 352 B/token vs 656 B `fp8_ds_mla` —
  **+38% KV pool** (415K → 571K tokens at equal settings) at decode parity, or the freed VRAM
  goes to the FP4 pool (the standing 4‑card config runs a 19.6 GiB/GPU pool + 175K‑token KV).

## DeepSeek‑V4‑Flash (159B)

Official checkpoint, 2‑bit experts + FP4 delta cache, MTP k=2, CUDA graphs (single‑stream
medians; prefill = 8k‑token prompt, uncached):

| hardware | decode | prefill 8k | context window |
|---|---:|---:|---:|
| **1× RTX PRO 6000 (96 GB)** | **161 tok/s** | **5 340 tok/s** | **512K** (needle PASS to 453K) |
| 2× RTX PRO 6000 (TP2) | 210 tok/s | 5 790 tok/s | 512K |
| 4× RTX 5090 (TP4) | 214 tok/s | 6 100 tok/s | KV‑bound† |
| **1× RTX 5090 (32 GB)** | **~38 tok/s** (host‑resident base) | — | KV‑bound† |

† Older revisions printed the 16K/8K *benchmark windows* here as if they were limits — they
are not. DS4's compressed KV is small (a single PRO 6000 reports **947K cached tokens**
alongside the full model — hence 512K windows on one card), and the 5090 configs have
gigabytes of KV headroom on the same math; a capacity + retrieval validation pass on the
5090s is queued and this table will carry its measured numbers.

**Batched serving** (aggregate decode tok/s at N concurrent streams; per‑stream in
parentheses at N=32):

| concurrency | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| 1× RTX PRO 6000 | 156 | 290 | 493 | 659 | **933** (29/stream) |
| 4× RTX 5090 (TP4) | 198 | 460 | 762 | 1 006 | **1 560** (49/stream) |

Four consumer 5090s match two PRO 6000s on decode. MTP acceptance ~2.6 tok/step across
configs. MTP also runs under **pipeline parallelism** (draft propagation + drafter embedding
share across ranks): DS4 on 4× RTX 5090 **PP4** does 184 tok/s vs 93 without (~2×), and greedy
decode under PP is **bit‑deterministic** (6/6 identical runs, with and without MTP).
Methodology: **[docs/v024-port.md](docs/v024-port.md)**.

---

## How it fits — 2‑bit experts at FP4 quality

We compress **only the routed experts** to 2 bits (the dense stack keeps the checkpoint's
precision — FP8 on DS4, NVFP4 on GLM) and recover FP4 precision adaptively:

- **2‑bit expert planes — the sign‑bias finding.** Naive 2‑bit *destroys* these models
  (degenerate loops). The cause is **sign asymmetry**, not error magnitude — the optimal‑L2
  codebook drops one sign's tail and the per‑expert bias compounds over dozens of layers.
  Forcing a **sign‑symmetric** `{−4,−1,1,4}` codebook at the same L2 error fixes it entirely
  (33,023 of 33,024 DS4 tensors pick it), landing MTP acceptance **at/above** the FP4 experts
  (2.73 ≥ 2.68 in the QUANT_PROBE study). The finding reproduces on **GLM‑5.2** (180‑tensor
  sweep: asym bias −0.042, 99% negative; symmetric 392× smaller at equal rel‑RMS).
- **FP4 recovery — used surgically.** Decode is HBM‑bound and an FP4 read is 2× the bytes, so
  2‑bit is the *fast* default: a **delta cache** keeps the hot experts at FP4 (background
  promote/evict, CUDA‑graph‑safe, `VLLM_MOE_W2_DELTA_GB=auto` sizes it from post‑KV VRAM),
  and a **confidence gate** (`VLLM_MOE_W2_GATE=1`) re‑runs low‑confidence steps at FP4 —
  force‑promote the step's routed experts, replay the graph once, re‑decide. Works inline on
  TP/single‑GPU (incl. MTP verify steps) and as a full‑pipeline replay under PP; τ tunable at
  runtime.
- **The kernels.** `moe_w2_mm` (2‑bit MoE GEMM: PRMT‑LUT in‑register decode → `QMMA.SF`
  block‑scaled tensor cores, 4 CTA/SM) and `moe_w4_mm` (FP4 delta GEMM) — hand‑written SASS,
  shipped as sources + prebuilt cubins for every sharding (K = 6144/4096/2048/1024/512), so
  TP2/TP4 work out of the box. Op‑validated (rel ~1–3e‑3, deterministic), graph‑capture‑exact.
  Prefill runs the **AFRAG** variant (fragment‑major activations → one `LDG.128` per QMMA
  A‑fragment; the prefill GEMM is load‑issue‑bound, not DRAM‑bound): bit‑identical outputs,
  1.3× on the GEMM, **+12% e2e prefill** on one card — default on (`VLLM_MOE_W2_AFRAG=0`
  opts out).

All three checkpoint flavors load: **FP4 experts** (DeepSeek‑V4‑Flash — codes remap),
**FP8 block‑quant** (Flash‑Base, GLM‑5.2‑FP8) and **modelopt NVFP4** (GLM‑5.2‑NVFP4) — the
latter two re‑quantized to the sign‑symmetric codebook at load, float64‑exact vs the
reference pipeline.

---

## When the model doesn't fit at all — the GPU as an expert cache

`VLLM_MOE_W2_BASE_CACHE_GB=N` inverts residency: the **whole 2‑bit base lives in pinned host
RAM**, and the GPU holds only the dense stack, KV, and an N‑GiB **cache of hot experts** (the
delta‑tier slot machinery, read inside CUDA graphs; background prefetch converges it to the
routed working set). MoE routing is concentrated enough to make this practical: **~19%
coverage serves ~96% of token→expert routings** on DS4, **~51% serves ~91%** on GLM —
measured live, not simulated.

Misses stay correct through the gate's replay trick: the desc kernel zeroes a missing
expert's contribution and bumps an in‑graph miss counter; the runner fetches **all** missing
routed experts in one batched pinned‑H2D transfer (51.6 GiB/s here; a 64‑expert fetch ≈ 3 ms)
and replays the step's graph once — **bit‑identical** to a fully resident forward
(unit‑tested). A **miss‑tolerance knob** (`VLLM_MOE_W2_BASE_MISS_TOL=k`, runtime‑tunable)
skips the replay when ≤ k of the step's ~600 routings miss — at 51% coverage that's ~2×
decode (21→44 tok/s on GLM TP2) with no systematic quality change observed (probes: PL
coherence, retrieval; quantitative eval pending).

Results: **DeepSeek‑V4‑Flash 159B on one RTX 5090** (72.7 GiB of 2‑bit planes vs 32 GB of
VRAM): ~38 tok/s steady with MTP (acceptance 2.83), coherent and deterministic.
**GLM‑5.2 753B on two RTX PRO 6000**: 45–52 tok/s with the full three‑tier stack (host 2‑bit
→ GPU 2‑bit cache → GPU FP4) — see the GLM table above. Neither model can otherwise run on
that hardware at any precision.

---

## The base: vLLM v0.24.0 on SM120

Upstream v0.24.0 ships DeepSeek‑V4 + GLM‑5.x + SM120 natively — but the release cannot
actually serve them on SM120. The patch carries the fixes (details in
[docs/v024-port.md](docs/v024-port.md)):

- **DeepGEMM**: release pin has no family‑120 host paths ("Unknown SF transformation",
  einsum/indexer asserts) → pin **nv‑dev `a6b593d2`** (as vLLM main did).
- **flashinfer**: official 0.6.12 pin predates the SM120 DS4 attention API → **0.6.14**.
- `cooperative_topk` uses thread‑block **cluster launch** (SM90/100‑only) → gated off on SM12x.
- o_proj fp8 einsum: SM100 packed scale layout NaNs on SM120 → SM90‑style raw f32 scales.
- CUDA‑graph capture: `thread_local` error mode on **all four** capture paths (the expert
  caches' background threads must not invalidate capture).

With the 2‑bit knobs off, the patch is exactly these base fixes — stock behaviour otherwise.

## Quickstart

```bash
git clone https://github.com/kacper-daftcode/vLLM-Moet && cd vLLM-Moet

# official vllm-openai:v0.24.0 image + patch + pins + SM120 cubins
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm120-v024 -t vllm-moet-sm120:v024 .
```

**GLM‑5.2 on 4× PRO 6000** (the standing agent‑serving config: 128K window, MTP, FP4 pool +
gate, tool/reasoning parsers):

```bash
docker run --rm --gpus '"device=0,1,2,3"' --network host --ipc host --shm-size 64g \
  -v /path/to/GLM-5.2-NVFP4:/model:ro \
  -e VLLM_MOE_W2=1 -e VLLM_MOE_W2_DELTA_GB=auto -e VLLM_MOE_W2_GATE=1 \
  vllm-moet-sm120:v024 \
  --model /model --served-model-name glm-5.2 --trust-remote-code \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --kv-cache-dtype fp8 --max-model-len 131072 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8000
```

(`--kv-cache-dtype nvfp4` enables the 352 B/token KV cache; it requires the FlashInfer JIT
patch from `tools/nvfp4_flashinfer_sm120/` baked into the image and is currently validated to
128K windows.)

**GLM‑5.2 on 2× PRO 6000** (three‑tier; needs ~200 GiB free host RAM for the pinned base):

```bash
docker run --rm --gpus '"device=0,1"' --network host --ipc host --shm-size 64g \
  -v /path/to/GLM-5.2-NVFP4:/model:ro \
  -e VLLM_MOE_W2=1 -e VLLM_MOE_W2_BASE_CACHE_GB=48 -e VLLM_MOE_W2_DELTA_GB=2 \
  -e VLLM_MOE_W2_GATE=1 -e VLLM_MOE_W2_BASE_MISS_TOL=8 \
  vllm-moet-sm120:v024 \
  --model /model --served-model-name glm-5.2 --trust-remote-code \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --kv-cache-dtype fp8 --max-model-len 16384 --kv-cache-memory-bytes 8589934592 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 --max-num-seqs 2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8000
```

**DeepSeek‑V4‑Flash on one PRO 6000** (161 tok/s, 512K window):

```bash
docker run --rm --gpus '"device=0"' --network host --ipc host --shm-size 64g \
  -v /path/to/DeepSeek-V4-Flash:/model:ro \
  -e VLLM_MOE_W2=1 -e VLLM_MOE_W2_DELTA_GB=1 \
  vllm-moet-sm120:v024 \
  --model /model --served-model-name deepseek-v4-flash --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len 24576 \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 1024 --max-num-seqs 4 \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --port 8000
```

`VLLM_MOE_W2=0` = stock FP4 path (needs ≥2 cards for DS4; GLM's stock NVFP4 does not fit this
box at all). TP: `--tensor-parallel-size 2|4` + `--disable-custom-all-reduce`. Single‑5090
DS4 (host‑resident base): `-e VLLM_MOE_W2_BASE_CACHE_GB=11 -e VLLM_MOE_W2_DELTA_GB=0`
(~80 GiB free host RAM; MTP works; PP unsupported on this path yet).

## Quality

Method: baseline is the untouched official checkpoint; our variant changes only the expert
codes (same stack, byte‑identical dense/scales/headers), so any delta is the quantization
alone — see [docs/quality.md](docs/quality.md). The QUANT_PROBE study (identical quant scheme
and cubins): MTP acceptance 2.73 vs 2.68 FP4 reference, draft accept 86.3% vs 84.1%, 12/12
coherent greedy outputs; bare 2‑bit agrees with FP4 on 89% of next‑token picks — the delta
cache + gate close that gap. Live serving reproduces the acceptance (~2.6 tok/step on DS4,
2.3–3.0 on GLM).

## The SM120 toolchain we built

These kernels exist only because we first built the assembler and the ISA data they need.
Consumer Blackwell (sm_120) has **no public SASS toolchain**. Current CUDA does expose the
block‑scaled MMA *instruction* itself (PTX `kind::mxf8f6f4` compiles to `QMMA.SF` — DeepGEMM's
SM120 port uses it), but everything these kernels are actually made of — hand scheduling
against measured latencies and control words, the PRMT‑LUT decode interleaved into the QMMA
stream, register‑bank and occupancy shaping (regcount 64 → 4 CTA/SM) — is decided by ptxas
and unreachable from CUDA/PTX. So the stack underneath this repo is end‑to‑end ours:

- **[`blackwell-isa`](https://github.com/kacper-daftcode/blackwell-isa)** — a machine‑readable
  **SM120 SASS ISA database**: 1,994 instruction forms, 128‑bit encoding templates + operand/
  bitfield maps, and per‑opcode scheduling metadata (pipeline/latency/throughput, control‑word
  classes). Reverse‑engineered and hardware‑validated on RTX 5090 (47,244 instructions decoded
  across 178 cubins at 100% coverage; 5,014/5,014 roundtrip‑fuzz). It documents what the CUDA
  toolchain hides — e.g. `QMMA.SF` block‑scaled FP4 MMA and an undocumented `E3M4` type code.
  Ships a [searchable HTML reference](https://kacper-daftcode.github.io/blackwell-isa/SM120_ISA_REFERENCE.html).
- **[`cubit`](https://github.com/kacper-daftcode/cubit)** — an **SM120 SASS assembler/disassembler**
  built on that database. It turns the hand‑written `.sass` sources in `kernels/sass/` into the
  cubins this server loads, and is the only tool needed to rebuild or audit them.

**ISA ([`blackwell-isa`](https://github.com/kacper-daftcode/blackwell-isa)) → assembler
([`cubit`](https://github.com/kacper-daftcode/cubit)) → SASS kernels → this vLLM.** None of the
kernels here can be *written* through stock CUDA on sm_120 — the instructions compile, the
kernels don't; this toolchain is what makes them possible.

## Repository layout
- **`patch/vllm-moet-v0.24.0.patch`** — the delta vs official vLLM `v0.24.0` (32 files,
  +5.6k lines; applies clean on the tag). Goes with the pins above.
- **`Dockerfile.sm120-v024`** — the image: official `vllm/vllm-openai:v0.24.0` + patch + pins +
  cubins.
- **`kernels/`** — SASS (`sass/`) + prebuilt SM120 cubins (`cubins-sm120/`, incl. the K=6144
  GLM‑5.x family) + generators (`gen/`) + `MANIFEST.md`.
- **`docs/v024-port.md`** — the port: pins, SM120 fixes, apply recipe, benchmark methodology.
- **`docs/quality.md`** — quality methodology.
