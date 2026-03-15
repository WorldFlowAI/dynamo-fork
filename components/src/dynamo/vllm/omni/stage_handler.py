# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic handler for a single vLLM-Omni disaggregated stage."""

import logging
import time
from typing import Any, AsyncGenerator, Dict

logger = logging.getLogger(__name__)


class OmniStageWorkerHandler:
    """Serves any single stage of any omni model.

    All behavior is driven by stage_config (from YAML). Not model-specific.
    Handles both stage-0 (direct input) and downstream stages (input via
    OmniConnector).
    """

    def __init__(
        self,
        engine,
        stage_config,
        connectors: dict[tuple[str, str], Any],
        stage_id: int,
    ):
        self.engine = engine
        self.stage_config = stage_config
        self.connectors = connectors
        self.stage_id = stage_id
        self.stage_type = getattr(stage_config, "stage_type", "llm")
        self.final_output = getattr(stage_config, "final_output", False)
        self.final_output_type = getattr(stage_config, "final_output_type", "text")
        self.engine_output_type = getattr(
            stage_config.engine_args, "engine_output_type", "text"
        )

    async def generate(
        self, request: Dict[str, Any], context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        request_id = request.get("request_id") or context.id()
        logger.debug(
            "OmniStageWorkerHandler stage=%d request=%s", self.stage_id, request_id
        )

        engine_inputs = self._resolve_inputs(request)
        if engine_inputs is None:
            yield {
                "error": f"Failed to resolve inputs for stage {self.stage_id}",
                "finished": True,
            }
            return

        sampling_params = request.get(
            "sampling_params",
            getattr(self.stage_config, "default_sampling_params", {}),
        )

        if self.stage_type == "diffusion":
            async for chunk in self._run_diffusion(
                engine_inputs, sampling_params, request_id
            ):
                yield chunk
        else:
            async for chunk in self._run_llm(
                engine_inputs, sampling_params, request_id
            ):
                yield chunk

    def _resolve_inputs(self, request: Dict[str, Any]) -> Any:
        """Resolve engine inputs from connector or direct request."""
        if request.get("from_connector"):
            from vllm_omni.distributed.omni_connectors.adapter import (
                try_recv_via_connector,
            )

            engine_inputs, rx_metrics = try_recv_via_connector(
                request, self.connectors, self.stage_id
            )
            if rx_metrics:
                logger.debug(
                    "Stage %d received %d bytes in %.1fms",
                    self.stage_id,
                    rx_metrics.get("rx_transfer_bytes", 0),
                    rx_metrics.get("rx_decode_time_ms", 0),
                )
            return engine_inputs
        else:
            return request.get("engine_inputs")

    async def _run_llm(
        self,
        engine_inputs: Any,
        sampling_params: Any,
        request_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run an AR (LLM) stage."""
        from vllm import SamplingParams as VllmSamplingParams

        sp_dict = dict(sampling_params) if sampling_params else {}
        sp = VllmSamplingParams(**sp_dict)

        # OmniLLM.generate() is synchronous and returns a list
        t0 = time.time()
        results = self.engine.generate(engine_inputs, sp, request_id=[request_id])
        elapsed = time.time() - t0

        logger.debug("Stage %d LLM generate took %.2fs", self.stage_id, elapsed)

        if not results:
            yield {"error": "No output from LLM engine", "finished": True}
            return

        result = results[0]

        # Stream final output if this stage produces user-visible output
        if self.final_output:
            yield self._format_final_output(result)

        # Always yield stage_output for the router to forward downstream
        yield {
            "stage_output": self._extract_stage_output(result),
            "finished": True,
        }

    async def _run_diffusion(
        self,
        engine_inputs: Any,
        sampling_params: Any,
        request_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run a diffusion stage."""
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        sp_dict = dict(sampling_params) if sampling_params else {}
        sp = OmniDiffusionSamplingParams(**sp_dict)

        t0 = time.time()
        results = self.engine.generate(engine_inputs, sp, request_ids=[request_id])
        elapsed = time.time() - t0

        logger.debug("Stage %d diffusion generate took %.2fs", self.stage_id, elapsed)

        if not results:
            yield {"error": "No output from diffusion engine", "finished": True}
            return

        result = results[0]

        if self.final_output:
            yield self._format_final_output(result)

        yield {
            "stage_output": self._extract_stage_output(result),
            "finished": True,
        }

    def _format_final_output(self, result) -> Dict[str, Any]:
        """Format output for client consumption."""
        output: Dict[str, Any] = {"final_output_type": self.final_output_type}

        if self.final_output_type == "image" and hasattr(result, "images"):
            output["images"] = result.images
        elif self.final_output_type == "text" and hasattr(result, "request_output"):
            ro = result.request_output
            if ro and ro.outputs:
                output["text"] = ro.outputs[0].text
        elif hasattr(result, "images"):
            output["images"] = result.images

        return {"final_data": output}

    def _extract_stage_output(self, result) -> Any:
        """Extract stage output for downstream forwarding via router."""
        # Return the full result object so the router's StageOutputProxy
        # can set it as engine_outputs for stage_input_processors
        return result

    def cleanup(self):
        """Clean up engine resources."""
        try:
            if hasattr(self.engine, "close"):
                self.engine.close()
            elif hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
            logger.info("Stage %d engine cleaned up", self.stage_id)
        except Exception as e:
            logger.error("Error cleaning up stage %d: %s", self.stage_id, e)
