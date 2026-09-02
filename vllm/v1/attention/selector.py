# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from typing import NamedTuple, cast, get_args

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    MAMBA_TYPE_TO_BACKEND_MAP,
    MambaAttentionBackendEnum,
)

logger = init_logger(__name__)


class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER
    use_non_causal: bool = False

    def __repr__(self):
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
            f"attn_type={self.attn_type}, "
            f"use_non_causal={self.use_non_causal})"
        )


def _uses_non_causal_dflash_attention(speculative_config) -> bool:
    if speculative_config is None or speculative_config.method != "dflash":
        return False

    # Path-oracle serving has two different attention roles under one
    # speculative config:
    #   * Target verifies the full tree (tree_width=2) and must use causal
    #     TREE_ATTN.
    #   * The copied draft config is forced to tree_width=1 by
    #     DFlashProposer._create_draft_vllm_config and must preserve the
    #     checkpoint's original bidirectional/non-causal DFlash attention.
    # Without this split the Target tries to instantiate TREE_ATTN with
    # use_non_causal=True and fails before the draft model is loaded.
    if (
        getattr(speculative_config, "enable_path_oracle", False)
        and getattr(speculative_config, "tree_width", 1) > 1
    ):
        return False

    head_type = getattr(speculative_config, "head_type", "auto")
    if head_type == "bidirectional":
        return True
    if head_type == "causal":
        return False

    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    dflash_config = getattr(hf_config, "dflash_config", None)
    if isinstance(dflash_config, dict):
        return not bool(dflash_config.get("causal_head", False))

    # Preserve the historical default for auto when no checkpoint metadata is
    # available: DFlash is treated as bidirectional/non-causal.
    return True


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    if kv_cache_dtype is not None:
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes, (
            f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
            f"Valid values are: {valid_cache_dtypes}"
        )

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    speculative_config = vllm_config.speculative_config
    use_non_causal = _uses_non_causal_dflash_attention(speculative_config)
    selected_backend = vllm_config.attention_config.backend
    if (
        speculative_config is not None
        and speculative_config.method == "dflash"
        and speculative_config.tree_width > 1
    ):
        selected_backend = AttentionBackendEnum.TREE_ATTN

    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type or AttentionType.DECODER,
        use_non_causal=use_non_causal,
    )

    return _cached_get_attn_backend(
        backend=selected_backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    from vllm.platforms import current_platform

    attention_cls = current_platform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    backend = resolve_obj_by_qualname(attention_cls)

    # Adjust kv cache layout if the selected backend requires a specific one
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
        logger.info(
            "Using %s KV cache layout for %s backend.",
            required_layout,
            backend.get_name(),
        )

    return backend


def get_mamba_attn_backend(
    mamba_type: str,
) -> type[AttentionBackend]:
    """Select which mamba attention backend to use and lazily import it."""
    return _cached_get_mamba_attn_backend(mamba_type)


@cache
def _cached_get_mamba_attn_backend(
    mamba_type: str,
) -> type[AttentionBackend]:
    assert mamba_type and isinstance(mamba_type, str)

    selected_backend = None
    try:
        backend_name = MAMBA_TYPE_TO_BACKEND_MAP[mamba_type]
        selected_backend = MambaAttentionBackendEnum[backend_name]
    except KeyError as e:
        raise ValueError(
            f"Invalid mamba attention backend type: '{mamba_type}'. Valid "
            f"types are: {list(MAMBA_TYPE_TO_BACKEND_MAP.keys())}"
        ) from e

    mamba_attn_backend = selected_backend.get_class()
    return mamba_attn_backend
