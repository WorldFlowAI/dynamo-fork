# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS model runner subclass for shadow mode.

Replaces monkey-patches with clean method overrides. Injected via
__class__ swap in GMSWorker.init_device().
"""

from __future__ import annotations

import logging
import time

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from gpu_memory_service.integrations.vllm.utils import is_shadow_mode

logger = logging.getLogger(__name__)


class GMSModelRunner(GPUModelRunner):
    """GPUModelRunner with shadow mode overrides.

    No __init__ or __slots__ — safe for __class__ swap.
    """

    def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        """No-op during shadow init; store config for later allocation on wake."""
        if getattr(self, "_shadow_init_phase", False):
            self._shadow_kv_cache_config = kv_cache_config
            self._shadow_kernel_block_sizes = kernel_block_sizes
            logger.info(
                "[Shadow] Init phase: stored config, skipping KV cache allocation"
            )
            return {}
        return super().initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)

    def _get_slot_mappings(self, *args, **kwargs):
        """Return (None, None) when KV caches are empty.

        _dummy_run() calls this unconditionally during warmup. Without KV
        tensors there is nothing to index into; (None, None) makes KV write
        ops gracefully no-op.
        """
        if not self.kv_caches:
            return None, None
        return super()._get_slot_mappings(*args, **kwargs)

    def _check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups):
        """In shadow mode, force PIECEWISE and skip backend resolution.

        vLLM's default resolution may escalate to FULL_AND_PIECEWISE which
        needs KV cache tensors for graph capture.
        """
        if is_shadow_mode():
            from vllm.config import CUDAGraphMode

            mode = self.compilation_config.cudagraph_mode
            if mode == CUDAGraphMode.NONE:
                # enforce_eager — keep NONE, just init keys
                self.cudagraph_dispatcher.initialize_cudagraph_keys(
                    CUDAGraphMode.NONE, self.uniform_decode_query_len
                )
            else:
                # Default shadow path — force PIECEWISE
                self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
                self.cudagraph_dispatcher.initialize_cudagraph_keys(
                    CUDAGraphMode.PIECEWISE, self.uniform_decode_query_len
                )
            return
        return super()._check_and_update_cudagraph_mode(
            attention_backends, kv_cache_groups
        )

    def allocate_kv_cache_on_wake(self) -> dict:
        """Allocate KV cache on wake using config stored during shadow init.

        Called by GMSWorker.wake_up() after _shadow_init_phase is cleared.
        Waits up to 60s for GPU memory to be freed.
        """
        assert hasattr(self, "_shadow_kv_cache_config"), (
            "_shadow_kv_cache_config not set"
        )
        assert hasattr(self, "_shadow_kernel_block_sizes"), (
            "_shadow_kernel_block_sizes not set"
        )

        config = self._shadow_kv_cache_config
        kv_cache_bytes = sum(t.size for t in config.kv_cache_tensors)

        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < kv_cache_bytes:
            logger.info(
                "[Shadow] Waiting for GPU memory (need %.2f GiB, free %.2f GiB)",
                kv_cache_bytes / (1 << 30),
                free_bytes / (1 << 30),
            )
            deadline = time.monotonic() + 60.0
            while free_bytes < kv_cache_bytes:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Timed out waiting for GPU memory: "
                        f"need {kv_cache_bytes / (1 << 30):.2f} GiB, "
                        f"free {free_bytes / (1 << 30):.2f} GiB"
                    )
                time.sleep(0.5)
                free_bytes = torch.cuda.mem_get_info()[0]
            logger.info(
                "[Shadow] GPU memory available (free %.2f GiB), proceeding",
                free_bytes / (1 << 30),
            )

        logger.info("[Shadow] Allocating KV cache on wake")

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(self.vllm_config):
            kv_caches = self.initialize_kv_cache_tensors(
                config,
                self._shadow_kernel_block_sizes,
            )

        # Re-register with KV transfer group (skipped at init since kv_caches was {}).
        # Mirrors GPUModelRunner.initialize_kv_cache() — update if upstream changes.
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                get_kv_transfer_group,
                has_kv_transfer_group,
            )

            if has_kv_transfer_group() and kv_caches:
                kv_transfer_group = get_kv_transfer_group()
                kv_transfer_group.register_kv_caches(kv_caches)
                logger.debug("[Shadow] Registered KV caches with transfer group")
        except ImportError:
            logger.debug("[Shadow] KV transfer group not available")

        total_bytes = sum(t.numel() * t.element_size() for t in kv_caches.values())
        logger.info(
            "[Shadow] Allocated KV cache on wake: %.2f GiB (%d tensors)",
            total_bytes / (1 << 30),
            len(kv_caches),
        )

        return kv_caches
