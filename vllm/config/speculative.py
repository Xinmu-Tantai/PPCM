# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field, SkipValidation, model_validator
from typing_extensions import Self

from vllm.config import LoadConfig
from vllm.config.kernel import MoEBackend
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import LazyLoader, has_arctic_inference

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)

MTPModelTypes = Literal[
    "deepseek_mtp",
    "mimo_mtp",
    "glm4_moe_mtp",
    "glm4_moe_lite_mtp",
    "glm_ocr_mtp",
    "ernie_mtp",
    "nemotron_h_mtp",
    "exaone_moe_mtp",
    "qwen3_next_mtp",
    "qwen3_5_mtp",
    "longcat_flash_mtp",
    "mtp",
    "pangu_ultra_moe_mtp",
    "step3p5_mtp",
]
NgramGPUTypes = Literal["ngram_gpu"]
DFlashModelTypes = Literal["dflash"]
EagleModelTypes = Literal[
    "eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes
]
SpeculativeMethod = Literal[
    "ngram",
    "medusa",
    "mlp_speculator",
    "draft_model",
    "suffix",
    EagleModelTypes,
    NgramGPUTypes,
]
RejectionSampleMethod = Literal["strict", "probabilistic", "synthetic"]


@config
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    enforce_eager: bool | None = None
    """Override the default enforce_eager from model_config"""
    # General speculative decoding control
    num_speculative_tokens: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: str | None = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: SpeculativeMethod | None = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""
    tensor_parallel_size: int | None = None
    """Users should pass "draft_tensor_parallel_size". This parameter's purpose is to
    warn users when they mistakenly provide the wrong argument."""

    # Draft model configuration
    quantization: me_quant.QuantizationMethods | str | None = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    moe_backend: MoEBackend | None = None
    """MoE backend to use for the draft model. When `None`, the draft model
    inherits the target model's `--moe-backend` setting. Useful when the
    drafter and generator require different MoE kernels (e.g. quantized
    generator with unquantized drafter)."""
    max_model_len: int | None = Field(default=None, ge=1)
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: str | None = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: str | None = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""

    # Advanced control
    disable_padded_drafter_batch: bool = False
    """Disable input padding for speculative decoding. If set to True,
    speculative input batches can contain sequences of different lengths,
    which may only be supported by certain attention backends. This currently
    only affects the EAGLE method of speculation."""
    use_local_argmax_reduction: bool = False
    """Use vocab-parallel local argmax instead of all-gathering full logits
    for draft token generation. Reduces communication from O(vocab_size) to
    O(2 * tp_size) per token. Only applies to greedy draft selection in
    non-tree speculation."""

    # Ngram proposer configuration
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""

    # Alternative drafting strategies
    speculative_token_tree: str | None = None
    """Specifies the tree structure for speculative token generation.
    """
    head_type: Literal["auto", "bidirectional", "causal"] = "auto"
    """Draft attention mode for DFlash. 'auto' uses the draft checkpoint's
    dflash_config.causal_head when available."""
    tree_width: int = Field(default=1, ge=1)
    """Requested draft tree width for DFlash tree inference experiments.
    Width 1 means linear drafting."""
    max_tree_budget: int | None = Field(default=None, ge=1)
    """Optional cap on the final root-inclusive DFlash verification tree."""
    tree_seed_budget: int | None = Field(default=None, ge=1)
    """Optional root-inclusive budget for the initially constructed DFlash
    seed tree. When larger than ``max_tree_budget``, the proposer builds this
    wider candidate tree first, then prunes it to the final verification
    budget by cumulative path log-probability. ``None`` preserves the original
    single-budget behavior."""
    tree_verify_budget: int | None = Field(default=None, ge=2)
    """PATR root-inclusive tree budget passed to Target verification."""
    enable_post_tree_head: bool = False
    """Enable the PPCM / PATR head (CCEL → CTIL → CPRL)."""
    post_tree_head_path: str | None = None
    """Directory containing the independently trained PATR safetensors."""
    post_tree_hidden_size: int = Field(default=512, ge=1)
    post_tree_num_layers: int = Field(default=2, ge=0)
    """Kept for checkpoint compatibility. The paper encoder is CCEL + CTIL + CPRL."""
    post_tree_num_heads: int = Field(default=8, ge=1)
    post_tree_num_kv_heads: int = Field(default=2, ge=1)
    post_tree_intermediate_size: int = Field(default=1024, ge=1)
    post_tree_candidate_only: bool = True
    post_tree_protect_spine: bool = False
    post_tree_compile: bool = False
    """Compile the loaded PATR head without changing its parameters/dtypes."""
    post_tree_cuda_graph: bool = False
    """Replay the eager PATR kernels with fixed-shape CUDA Graph buffers."""
    post_tree_tree_width: int | None = Field(default=None, ge=1)
    """Branching factor used by PATR rank embeddings.

    Draft model creation forces ``tree_width=1`` so the draft stack stays
    linear. When that happens, this field keeps the real tree-draft width
    (e.g. 7) for PATR construction / checkpoint loading.
    """
    enable_path_selector: bool = False
    """Enable the fixed 7-step, Top-2 Path-Head-Mask selector."""
    path_selector_path: str | None = None
    """Directory containing the independently trained path-selector weights."""
    path_selector_stats_path: str | None = None
    """If set, append one JSONL row per draft after target verify. This is
    passed through EngineCore config because worker processes may not inherit
    PATH_SELECTOR_STATS_PATH from the API server environment."""
    enable_dflash2_beam_selector: bool = False
    """Select one chain from a Top-16 DFlash2 lattice with beam-16 and a
    learned 16-head path reranker."""
    dflash2_beam_selector_path: str | None = None
    """Directory containing DFlash2 beam-selector config and safetensors."""
    enable_path_oracle: bool = False
    """Evaluation-only mode that verifies the complete fixed 7-step Top-2
    prefix tree. This measures the candidate-set oracle acceptance upper bound;
    it is not a deployable selector because Target evaluates all 255 nodes."""
    tree_draft: Literal[
        "accum_logp", "entropy", "hybrid", "opt_prefix", "top2gap_fanout",
    ] = "accum_logp"
    """Scoring strategy for DFlash tree node expansion.
    'accum_logp' prioritises high-probability prefixes (original behaviour).
    'entropy' prioritises uncertain positions (high per-depth entropy).
    'hybrid' combines cumulative log-prob with entropy (weighted by
    tree_hybrid_alpha).
    'opt_prefix' builds the provably optimal tree under factorized draft
    marginals by selecting the top-B prefix-probability nodes via a best-first
    heap (DDTree algorithm).  Ignores tree_construction.
    'top2gap_fanout' caps per-depth fanout from the rank-1/rank-2 logprob gap
    and uses cumulative-log-prob heap expansion."""
    tree_hybrid_alpha: float = Field(default=1.0, gt=0.0)
    """Weight applied to per-depth entropy in 'hybrid' scoring mode.
    Larger values shift budget towards uncertain positions."""
    max_draft_passes: int = Field(default=0, ge=0)
    """Number of prune/regrow refinement passes after the initial tree is
    built.  Applies to any tree_draft mode; set to 0 to disable."""
    tree_prune_ratio: float = Field(default=0.25, gt=0.0, lt=1.0)
    """Fraction of leaves to prune in each refinement pass.
    Only used when max_draft_passes > 0."""
    tree_construction: Literal["depth_first", "breadth_first"] = "breadth_first"
    """Tree node allocation strategy. 'depth_first' pre-allocates the greedy
    (top-1) spine to full depth before spending budget on side branches,
    guaranteeing tree acceptance >= linear-chain acceptance. 'breadth_first'
    uses best-cumulative-log-prob heap expansion (legacy behaviour)."""
    tree_attn_kernel: Literal["triton", "optimus"] = "triton"
    """Attention kernel for DFlash tree verification. 'triton' uses the
    default Triton bias-based path; 'optimus' uses the fused SM90 paged
    tree-mask kernel from optimus_cutedsl (requires SM90 GPU and
    page_size == 128)."""
    tree_kv_layout: Literal["physical", "logical"] = "physical"
    """KV-cache layout for DFlash tree verification. 'physical' keeps the
    current compact-after-accept path; 'logical' tracks accepted tree nodes via
    slot indirection instead of moving KV entries."""
    num_cudagraph_tree_captures: int = Field(default=0, ge=0)
    """Enable DFlash tree CUDAGraph capture.

    When > 0, tree verification captures only max-budget tree shapes
    (dflash_tree_budget * num_reqs).  Trees are adjusted to dflash_tree_budget
    before verification so every target forward is a CUDAGraph hit.  0 disables
    DFlash tree CUDAGraph capture.
    """
    parallel_drafting: bool = False
    """Enable parallel drafting, where all speculative tokens are generated
    in parallel rather than sequentially. This can improve performance but
    requires the speculative model be trained to support parallel drafting.
    Only compatible with EAGLE and draft model methods."""

    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

    # Suffix decoding configuration
    suffix_decoding_max_tree_depth: int = 24
    """The maximum depth of the suffix decoding global and prompt trees. The
    tree depth limits the sum of the prefix match and speculation lengths."""

    suffix_decoding_max_cached_requests: int = 10000
    """The maximum number of requests to cache in the global suffix tree. If
    exceeded, will trigger eviction in FIFO order. If set to 0, the global
    suffix tree is disabled and past responses are not cached (prompt trees
    are still used)."""

    suffix_decoding_max_spec_factor: float = 1.0
    """The maximum spec factor for suffix decoding. The spec factor controls
    speculation lengths based on the prefix match length: max_spec_tokens =
    max_spec_factor * prefix_match_length."""

    suffix_decoding_min_token_prob: float = 0.1
    """The minimum token probability for suffix decoding. Will only speculate
    tokens with estimated probability (based on frequency counts) greater than
    or equal to this value."""

    draft_load_config: LoadConfig | None = None
    """Load config for the draft model. If not specified, will use the load
    config from the target model."""

    rejection_sample_method: RejectionSampleMethod = "strict"
    """Whether to use strict (target and draft sampled tokens match exactly)
    or probabilistic rejection sampling. Both respect the target model
    distribution, but the latter yields a higher acceptance rate at the cost
    of more memory to cache draft logits."""

    synthetic_acceptance_rate: float | None = None
    """Average acceptance rate for synthetic rejection sampling. Draft
    tokens are accepted with a position-dependent probability that decays
    geometrically, calibrated so that the mean rate across all speculative
    positions equals this value. Only used when rejection_sample_method
    is 'synthetic'. Must be in [0, 1]."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        # Eagle3 and extract_hidden_states affect the computation graph because
        # they return intermediate hidden states in addition to the final hidden state.
        uses_aux_hidden_states = self.method in (
            "eagle3",
            "extract_hidden_states",
            "dflash",
        )
        factors.append(uses_aux_hidden_states)

        # The specific layers used also affect the computation graph
        if uses_aux_hidden_states and self.draft_model_config is not None:
            layer_ids = getattr(
                self.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                None,
            )
            if layer_ids is not None:
                # Convert to tuple to make it hashable
                factors.append(tuple(layer_ids))
        if self.method == "dflash":
            factors.extend(
                (
                    self.head_type,
                    self.tree_width,
                    self.max_tree_budget,
                    self.tree_seed_budget,
                    self.tree_verify_budget,
                    self.enable_post_tree_head,
                    self.post_tree_hidden_size,
                    self.post_tree_num_layers,
                    self.post_tree_num_heads,
                    self.post_tree_num_kv_heads,
                    self.post_tree_intermediate_size,
                    self.post_tree_candidate_only,
                    self.post_tree_protect_spine,
                    self.post_tree_compile,
                    self.post_tree_cuda_graph,
                    self.post_tree_tree_width,
                    self.enable_path_selector,
                    self.enable_dflash2_beam_selector,
                    self.enable_path_oracle,
                    self.tree_draft,
                    self.tree_hybrid_alpha,
                    self.max_draft_passes,
                    self.tree_attn_kernel,
                    self.tree_kv_layout,
                    self.num_cudagraph_tree_captures,
                )
            )

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
            hf_config.model_type = "glm4_moe_lite_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeLiteMTPModel"],
                }
            )

        if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
            hf_config.model_type = "glm_ocr_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["GlmOcrMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if (
            hf_config.model_type in {"nemotron_h", "nemotron_h_puzzle"}
            and hasattr(hf_config, "num_nextn_predict_layers")
            and hf_config.num_nextn_predict_layers > 0
        ):
            # Check if this is an MTP variant
            hf_config.model_type = "nemotron_h_mtp"
        if hf_config.model_type == "nemotron_h_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )

        if hf_config.model_type == "exaone_moe":
            hf_config.model_type = "exaone_moe_mtp"
        if hf_config.model_type == "exaone_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]}
            )

        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            is_moe = hf_config.model_type == "qwen3_5_moe"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        if hf_config.model_type == "step3p5":
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        return hf_config

    def __post_init__(self):
        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        # infer method from user args
        if self.method is None:
            if self.model in ("ngram", "[ngram]"):
                self.method = "ngram"
            else:
                self.method = "draft_model"

        if self.method in get_args(MTPModelTypes) and self.method != "mtp":
            logger.warning(
                "method `%s` is deprecated and replaced with mtp.", self.method
            )
            self.method = "mtp"

        if self.model is None and self.num_speculative_tokens is not None:
            if self.method == "mtp":
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for mtp")
                if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
                    # FIXME(luccafong): cudagraph with v32 MTP is not supported,
                    # remove this when the issue is fixed.
                    self.enforce_eager = True
                # use the draft model from the same model:
                self.model = self.target_model_config.model
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            elif self.method == "ngram_gpu":
                self.model = "ngram_gpu"
            elif self.method == "suffix":
                self.model = "suffix"
            elif self.method == "extract_hidden_states":
                self.model = "extract_hidden_states"
            else:
                raise ValueError(
                    "num_speculative_tokens was provided but without speculative model."
                )

        if self.method in ("ngram", "[ngram]"):
            self.method = "ngram"

        if self.method in ("ngram", "ngram_gpu"):
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "suffix":
            self._validate_suffix_decoding()
        elif self.method == "extract_hidden_states":
            from vllm.transformers_utils.configs.extract_hidden_states import (
                ExtractHiddenStatesConfig,
            )

            # ExtractHiddenStatesModel is instantiated manually in load_model()
            # We just need to store the target model config for KV cache shape info
            self.model = "extract_hidden_states"
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if hasattr(self.draft_model_config, "hf_config"):
                hf_config = self.draft_model_config.hf_config.to_dict()
            elif (
                isinstance(self.draft_model_config, dict)
                and "hf_config" in self.draft_model_config
            ):
                hf_config = self.draft_model_config["hf_config"]
            else:
                hf_config = {}

            self.draft_model_config = copy.copy(self.target_model_config)
            self.draft_model_config.hf_config = ExtractHiddenStatesConfig(
                self.draft_model_config.hf_config, **hf_config
            )
            self.update_arch_()
            self.draft_parallel_config = self.target_parallel_config

        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.trust_remote_code,
                    allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                    config_format=self.target_model_config.config_format,
                )

                # Automatically detect the method
                if self.method in ("eagle", "eagle3", "dflash"):
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                elif "eagle-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    self.method = "eagle3"
                elif "dflash" in self.draft_model_config.model.lower():
                    self.method = "dflash"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run "
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.draft_model_config.hf_config.model_type in (
                    "longcat_flash_mtp"
                ):
                    self.method = "longcat_flash_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "LongCat MTP models only have "
                            "one layer. Might need some code changes "
                            "to support multiple layers."
                        )
                elif self.method == "draft_model":
                    pass
                else:
                    raise NotImplementedError(
                        f"Unsupported speculative method: '{self.method}'"
                    )

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3", "dflash"):
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig
                    from vllm.transformers_utils.configs.speculators import (
                        SpeculatorsConfig,
                    )

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config
                        self.update_arch_()

                if self.method == "dflash":
                    self.parallel_drafting = True
                    if self.tree_width > 1 and self.head_type == "bidirectional":
                        raise ValueError(
                            "Native DFlash tree drafting only supports causal heads."
                        )

                if self.num_speculative_tokens is not None and hasattr(
                    self.draft_model_config.hf_config, "num_lookahead_tokens"
                ):
                    self.draft_model_config.hf_config.num_lookahead_tokens = (
                        self.num_speculative_tokens
                    )

                n_predict = getattr(
                    self.draft_model_config.hf_config, "n_predict", None
                )
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}"
                        )

                if self.speculative_token_tree is None:
                    if self.num_speculative_tokens is None:
                        raise ValueError(
                            "A speculative model was provided, but neither "
                            "`speculative_token_tree` nor `num_speculative_tokens` "
                            "was provided"
                        )

                    # Generate chain of tokens.
                    self.speculative_token_tree = str(
                        [(i + 1) * (0,) for i in range(self.num_speculative_tokens)]
                    )
                else:
                    # Sort the token tree breadth-first.
                    tree_choices = ast.literal_eval(self.speculative_token_tree)
                    self.speculative_token_tree = str(
                        sorted(tree_choices, key=lambda t: (len(t), t))
                    )

                self.draft_tensor_parallel_size = (
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config,
                    )
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    )
                )

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )
        return self

    def _validate_suffix_decoding(self):
        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference==0.1.1`."
            )
        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )
        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: int | None,
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {draft_max_model_len=}"
                )

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {target_max_model_len=}"
                )

            return speculative_max_model_len

        return min(
            draft_max_model_len,
            target_max_model_len,
        )

    @staticmethod
    def _verify_and_get_draft_tp(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int | None,
        draft_hf_config: PretrainedConfig,
    ) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type,
                    )
            else:
                speculative_draft_tensor_parallel_size = (
                    target_parallel_config.tensor_parallel_size
                )
        elif speculative_draft_tensor_parallel_size not in (
            1,
            target_parallel_config.tensor_parallel_size,
        ):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size"
            )
        return speculative_draft_tensor_parallel_size

    def update_arch_(self):
        """
        EagleConfig and ExtractHiddenStatesConfig update architectures, so update all
        architectures-related fields in self.draft_model_config
        """
        self.draft_model_config.hf_text_config = get_hf_text_config(
            self.draft_model_config.hf_config
        )
        self.draft_model_config.model_arch_config = (
            self.draft_model_config.get_model_arch_config()
        )
        model_info, arch = self.draft_model_config.registry.inspect_model_cls(
            self.draft_model_config.architectures,
            self.draft_model_config,
        )
        self.draft_model_config._model_info = model_info
        self.draft_model_config._architecture = arch

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if self.tensor_parallel_size is not None:
            raise ValueError(
                "'tensor_parallel_size' is not a valid argument in the "
                "speculative_config. Please pass 'draft_tensor_parallel_size' instead."
            )

        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter."
            )

        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "Expected num_speculative_tokens to be greater "
                f"than zero ({self.num_speculative_tokens})."
            )

        if (
            self.method == "dflash"
            and self.tree_width > 1
            and self.tree_seed_budget is not None
            and self.tree_seed_budget < self.dflash_tree_budget
        ):
            raise ValueError(
                "tree_seed_budget must be greater than or equal to the final "
                "DFlash verification budget (max_tree_budget). Got "
                f"tree_seed_budget={self.tree_seed_budget}, "
                f"final_budget={self.dflash_tree_budget}."
            )

        if self.method == "dflash" and self.enable_post_tree_head:
            if self.post_tree_compile and self.post_tree_cuda_graph:
                raise ValueError(
                    "post_tree_compile and post_tree_cuda_graph are mutually "
                    "exclusive. The validated exact path uses CUDA Graph only."
                )
            if self.tree_seed_budget is None or self.tree_verify_budget is None:
                raise ValueError(
                    "PATR requires explicit tree_seed_budget and "
                    "tree_verify_budget."
                )
            if self.tree_verify_budget > self.tree_seed_budget:
                raise ValueError(
                    "tree_verify_budget must not exceed tree_seed_budget."
                )
            if self.tree_draft != "accum_logp":
                raise ValueError(
                    "PATR currently supports tree_draft='accum_logp' only."
                )
            if self.max_draft_passes != 0:
                raise ValueError("PATR and max_draft_passes are mutually exclusive.")
            if not self.post_tree_candidate_only:
                raise ValueError(
                    "PATR currently supports candidate-only scoring only."
                )
            if not self.post_tree_head_path:
                raise ValueError("PATR requires post_tree_head_path.")
            if not Path(self.post_tree_head_path).is_dir():
                raise ValueError(
                    "PATR checkpoint directory does not exist: "
                    f"{self.post_tree_head_path}"
                )
            if self.post_tree_hidden_size % self.post_tree_num_heads != 0:
                raise ValueError(
                    "post_tree_hidden_size must be divisible by "
                    "post_tree_num_heads."
                )
            if self.post_tree_num_heads % self.post_tree_num_kv_heads != 0:
                raise ValueError(
                    "post_tree_num_heads must be divisible by "
                    "post_tree_num_kv_heads."
                )
            if self.post_tree_protect_spine:
                if self.tree_construction != "depth_first":
                    raise ValueError(
                        "A complete protected PATR spine requires "
                        "tree_construction='depth_first'."
                    )
                if self.tree_verify_budget < self.num_speculative_tokens + 1:
                    raise ValueError(
                        "tree_verify_budget must fit root plus the complete "
                        "greedy spine."
                    )

        if self.method == "dflash" and self.enable_path_selector:
            if self.enable_post_tree_head:
                raise ValueError(
                    "DFlash path selection and PATR are mutually exclusive."
                )
            if self.num_speculative_tokens != 7:
                raise ValueError(
                    "DFlash path selection is fixed to 7 speculative tokens."
                )
            if self.tree_width != 1:
                raise ValueError(
                    "DFlash path selection verifies one selected chain and "
                    "therefore requires tree_width=1."
                )
            if self.max_draft_passes != 0:
                raise ValueError(
                    "DFlash path selection and max_draft_passes are mutually "
                    "exclusive."
                )
            if not self.path_selector_path:
                raise ValueError(
                    "DFlash path selection requires path_selector_path."
                )
            if not Path(self.path_selector_path).is_dir():
                raise ValueError(
                    "DFlash path-selector checkpoint directory does not exist: "
                    f"{self.path_selector_path}"
                )

        if self.method == "dflash" and self.enable_dflash2_beam_selector:
            if self.enable_path_selector or self.enable_post_tree_head:
                raise ValueError(
                    "DFlash2 beam selection, fixed path selection, and PATR "
                    "are mutually exclusive."
                )
            if self.enable_path_oracle:
                raise ValueError(
                    "DFlash2 beam selection and path oracle are mutually exclusive."
                )
            if self.num_speculative_tokens != 7 or self.tree_width != 1:
                raise ValueError(
                    "DFlash2 beam selection requires 7 speculative tokens and "
                    "tree_width=1."
                )
            if self.max_draft_passes != 0:
                raise ValueError(
                    "DFlash2 beam selection requires max_draft_passes=0."
                )
            path = self.dflash2_beam_selector_path
            if not path or not Path(path).is_dir():
                raise ValueError(
                    "DFlash2 beam-selector checkpoint directory does not exist: "
                    f"{path}"
                )

        if self.method == "dflash" and self.enable_path_oracle:
            if (
                self.enable_path_selector
                or self.enable_dflash2_beam_selector
                or self.enable_post_tree_head
            ):
                raise ValueError(
                    "DFlash path oracle, path selector, and PATR are mutually "
                    "exclusive."
                )
            if self.num_speculative_tokens != 7:
                raise ValueError(
                    "DFlash path oracle is fixed to 7 speculative tokens."
                )
            if self.tree_width != 2:
                raise ValueError(
                    "DFlash path oracle requires tree_width=2."
                )
            if self.dflash_tree_budget != 255:
                raise ValueError(
                    "DFlash path oracle must verify the complete root-inclusive "
                    "Top-2 tree (max_tree_budget=255)."
                )
            if self.max_draft_passes != 0:
                raise ValueError(
                    "DFlash path oracle and max_draft_passes are mutually "
                    "exclusive."
                )

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )

        aux_hidden_states_supported = [
            "llama",
            "qwen",
            "minicpm",
            "gpt_oss",
            "hunyuan_vl",
            "hunyuan_v1_dense",
            "afmoe",
            "nemotron_h",
            "deepseek_v2",
            "deepseek_v3",
            "kimi_k2",
            "kimi_k25",
            "step3p5",
        ]
        if (
            self.method in ("eagle3", "extract_hidden_states", "dflash")
            and self.target_model_config
            and not any(
                supported_model in self.target_model_config.hf_text_config.model_type
                for supported_model in aux_hidden_states_supported
            )
        ):
            raise ValueError(
                f"{self.method} is only supported for {aux_hidden_states_supported}"
                f" models. Got {self.target_model_config.hf_text_config.model_type=}"
            )
        self.verify_equal_vocab_size_if_draft_model()
        return self

    def verify_equal_vocab_size_if_draft_model(self):
        if (
            self.method == "draft_model"
            and self.target_model_config is not None
            and self.draft_model_config is not None
        ):
            target_vocab_size = self.target_model_config.get_vocab_size()
            draft_vocab_size = self.draft_model_config.get_vocab_size()
            if target_vocab_size != draft_vocab_size:
                raise ValueError(
                    f"Target and draft model should have the same vocabulary size. "
                    f"Target model vocab_size={target_vocab_size}. "
                    f"Draft model vocab_size={draft_vocab_size}. "
                    f"Using models with different tokenizers can cause out-of-bounds "
                    f"errors during speculative decoding."
                )

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """
        Calculate the maximum number of new slots that might be added to the batch
        when drafting.
        """
        slots_per_req = 0  # for serial non-draft-model methods, no change needed
        if self.method == "dflash" and self.tree_width > 1:
            tree_budget = self.dflash_tree_budget
            # Tree verification adds one slot per speculative tree node beyond the
            # already-sampled root token.
            slots_per_req = max(tree_budget - 1, 0)
        elif self.parallel_drafting:
            # For parallel drafting, we need one new slot per 'masked' token
            slots_per_req = self.num_speculative_tokens - 1
        if self.uses_draft_model():
            # For draft model-based speculation, we need one new slot per request
            # Since we do not slice the draft tokens
            slots_per_req += 1
        return slots_per_req

    @property
    def dflash_tree_budget(self) -> int:
        tree_block_size = self.num_speculative_tokens + 1
        full_tree_size = (self.tree_width**tree_block_size - 1) // (
            self.tree_width - 1
        )
        if self.enable_post_tree_head:
            assert self.tree_verify_budget is not None
            return min(full_tree_size, self.tree_verify_budget)
        if self.max_tree_budget is not None and self.max_tree_budget > 0:
            return min(full_tree_size, self.max_tree_budget)
        return full_tree_size

    @property
    def dflash_seed_tree_budget(self) -> int:
        """Root-inclusive candidate-tree budget before optional pruning."""
        if self.tree_seed_budget is None:
            return self.dflash_tree_budget
        tree_block_size = self.num_speculative_tokens + 1
        full_tree_size = (self.tree_width**tree_block_size - 1) // (
            self.tree_width - 1
        )
        return min(full_tree_size, self.tree_seed_budget)

    @property
    def cudagraph_tree_capture_sizes(self) -> list[int] | None:
        """Per-request tree sizes for DFlash CUDAGraph capture.

        Returns ``None`` when the feature is disabled
        (``num_cudagraph_tree_captures == 0`` or non-tree mode).
        When enabled, captures only the maximum tree budget.  Target
        verification CUDA graphs then cover ``dflash_tree_budget * num_reqs``
        for each request count.
        """
        if self.num_cudagraph_tree_captures <= 0:
            return None
        if self.method != "dflash" or self.tree_width <= 1:
            return None
        return [self.dflash_tree_budget]

    @property
    def cudagraph_uniform_decode_query_len(self) -> int:
        if self.method == "dflash" and self.tree_width > 1:
            return self.dflash_tree_budget
        return 1 + self.num_speculative_tokens

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp", "dflash")

    def use_dflash(self) -> bool:
        return self.method == "dflash"

    def uses_draft_model(self) -> bool:
        return self.method == "draft_model"

    def uses_extract_hidden_states(self) -> bool:
        return self.method == "extract_hidden_states"

    def use_ngram_gpu(self) -> bool:
        return self.method == "ngram_gpu"

    def __repr__(self) -> str:
        method = self.method
        model = (
            None
            if method in ("ngram", "suffix", "extract_hidden_states")
            else self.draft_model_config.model
        )
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
