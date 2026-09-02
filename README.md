# PPCM

**Parallel Path Causal Modeling** for speculative decoding.

PPCM takes the Top-7 draft candidates and jointly models adjacent positions with a three-layer encoder:

1. **CCEL** — Causal Context Encoding Layer  
2. **CTIL** — Candidate Token Interaction Layer  
3. **CPRL** — Causal Path Refinement Layer  

This repository is a vLLM serving stack with PPCM enabled. The released checkpoint packs a 5-layer causal draft head and the PPCM encoder into one set of weights.

| | Link |
|---|---|
| Code | [github.com/Xinmu-Tantai/PPCM](https://github.com/Xinmu-Tantai/PPCM) |
| Weights | [huggingface.co/Xinmu7/PPCM](https://huggingface.co/Xinmu7/PPCM) |

## Model

The public checkpoint is **Qwen3-8B-PPCM** (`PPCMDraftModel`):

- Target: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- Draft: 5-layer causal draft, hidden size and vocabulary aligned with Qwen3-8B
- PPCM: CCEL → CTIL → CPRL over Top-7 candidates
- Speculative length: **7** (`block_size=8`)
- Tree width: **7**

Download the weights from [Xinmu7/PPCM](https://huggingface.co/Xinmu7/PPCM). The Target Qwen3-8B checkpoint is not included and must be loaded separately.

## Usage

Point the draft / PPCM path at the Hugging Face repo (or a local copy) and run with **7 speculative tokens**:

```bash
# Target: Qwen/Qwen3-8B
# Draft + PPCM: Xinmu7/PPCM
# NUM_SPECULATIVE_TOKENS=7
```

Example scripts live under `examples/offline_inference/` (`ppcm_profiling_*.sh`). Override:

```bash
TARGET_MODEL=/path/to/Qwen3-8B
DRAFT_MODEL=/path/to/Qwen3-8B-PPCM
```

This tree is source-only. Native CUDA extensions are not shipped here; build them from this repository before serving.

## Code map

- PPCM encoder: `vllm/model_executor/models/dflash_tree_post_head.py` (`CCEL` → `CTIL` → `CPRL`)
- Draft + PPCM loader: `vllm/model_executor/models/qwen3_dflash.py` (reads draft tensors and `ppcm.*` from the same `model.safetensors`)
- Hugging Face custom class: `ppcm.py` in [Xinmu7/PPCM](https://huggingface.co/Xinmu7/PPCM) (`PPCMDraftModel`)

## License

Apache License 2.0. This project is built on [vLLM](https://github.com/vllm-project/vllm).
