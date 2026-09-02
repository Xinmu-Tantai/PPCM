# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_tree_drafts: int = 0
    num_tree_nodes: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_tree_nodes_per_depth: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_tree_nodes_per_depth=[0] * num_spec_tokens,
        )

    def observe_draft(
        self,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        tree_size: int = 0,
        tree_nodes_per_depth: list[int] | None = None,
    ):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        if tree_size > 0:
            self.num_tree_drafts += 1
            self.num_tree_nodes += tree_size
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
        if tree_nodes_per_depth is not None:
            for i, cnt in enumerate(tree_nodes_per_depth):
                if i < len(self.num_tree_nodes_per_depth):
                    self.num_tree_nodes_per_depth[i] += cnt


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.num_tree_drafts: list[int] = []
        self.num_tree_nodes: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.tree_nodes_per_depth_lists: list[list[int]] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.num_tree_drafts.append(spec_decoding_stats.num_tree_drafts)
        self.num_tree_nodes.append(spec_decoding_stats.num_tree_nodes)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )
        self.tree_nodes_per_depth_lists.append(
            spec_decoding_stats.num_tree_nodes_per_depth
        )

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        num_tree_drafts = np.sum(self.num_tree_drafts)
        num_tree_nodes = np.sum(self.num_tree_nodes)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        tree_shape_str = ""
        if self.tree_nodes_per_depth_lists and num_tree_drafts > 0:
            depth_matrix = np.array(self.tree_nodes_per_depth_lists)
            avg_per_depth = np.sum(depth_matrix, axis=0) / num_tree_drafts
            tree_shape_str = ", ".join(f"{v:.1f}" for v in avg_per_depth)

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%"
            + (", Avg tree nodes per depth: %s" if tree_shape_str else ""),
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
            *([tree_shape_str] if tree_shape_str else []),
        )
        self.reset()


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        counter_drafts = self._counter_cls(
            name="vllm:spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_drafts = create_metric_per_engine(
            counter_drafts, per_engine_labelvalues
        )

        counter_draft_tokens = self._counter_cls(
            name="vllm:spec_decode_num_draft_tokens",
            documentation="Number of draft tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_draft_tokens = create_metric_per_engine(
            counter_draft_tokens, per_engine_labelvalues
        )

        counter_accepted_tokens = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens",
            documentation="Number of accepted tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens = create_metric_per_engine(
            counter_accepted_tokens, per_engine_labelvalues
        )

        counter_tree_drafts = self._counter_cls(
            name="vllm:spec_decode_num_tree_drafts",
            documentation="Number of DFlash tree verification rounds.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_tree_drafts = create_metric_per_engine(
            counter_tree_drafts, per_engine_labelvalues
        )

        counter_tree_nodes = self._counter_cls(
            name="vllm:spec_decode_num_tree_nodes",
            documentation="Number of DFlash tree nodes verified.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_tree_nodes = create_metric_per_engine(
            counter_tree_nodes, per_engine_labelvalues
        )

        assert speculative_config is not None
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if self.spec_decoding_enabled
            else 0
        )
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

        depth_labelnames = labelnames + ["depth"]
        base_depth_counter = self._counter_cls(
            name="vllm:spec_decode_num_tree_nodes_per_depth",
            documentation="Tree nodes drafted per depth level.",
            labelnames=depth_labelnames,
        )
        self.counter_spec_decode_num_tree_nodes_per_depth: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_depth_counter.labels(*lv, str(d)) for d in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        self.counter_spec_decode_num_tree_drafts[engine_idx].inc(
            spec_decoding_stats.num_tree_drafts
        )
        self.counter_spec_decode_num_tree_nodes[engine_idx].inc(
            spec_decoding_stats.num_tree_nodes
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
        for depth, counter in enumerate(
            self.counter_spec_decode_num_tree_nodes_per_depth[engine_idx]
        ):
            if depth < len(spec_decoding_stats.num_tree_nodes_per_depth):
                counter.inc(spec_decoding_stats.num_tree_nodes_per_depth[depth])
