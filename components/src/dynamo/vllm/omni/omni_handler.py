# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, Optional, Union

from diffusers.utils import export_to_video
from fsspec.implementations.dirfs import DirFileSystem
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.protocols.audio_protocol import (
    AudioData,
    NvAudioSpeechResponse,
    NvCreateAudioSpeechRequest,
)
from dynamo.common.protocols.image_protocol import (
    ImageData,
    NvCreateImageRequest,
    NvImagesResponse,
)
from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)
from dynamo.common.storage import upload_to_fs
from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.common.utils.video_utils import (
    compute_num_frames,
    normalize_video_frames,
    parse_size,
)
from dynamo.vllm.omni.base_handler import BaseOmniHandler

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_FPS = 16

# TTS constants (matching vLLM-Omni serving_speech.py)
# model_stage names that receive Qwen3-TTS-specific prompt format
# (prompt_token_ids + additional_information). Other audio models
# (MiMo-Audio, Qwen3-Omni, Stable Audio, etc.) use a plain text prompt.
_TTS_MODEL_STAGES: set = {"qwen3_tts"}
_TTS_LANGUAGES = {
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
}
_TTS_MAX_INSTRUCTIONS_LENGTH = 500
_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096
_REF_AUDIO_TIMEOUT_S = 15
_REF_AUDIO_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass
class EngineInputs:
    """Parsed engine inputs ready for AsyncOmni.generate().

    Attributes:
        prompt: OmniTextPrompt dict for the engine.
        sampling_params_list: Per-stage sampling parameters, or None for defaults.
        request_type: The resolved request type (may differ from the initial parse
            when a chat completion request carries video params).
        fps: Frames per second, only meaningful for video requests.
        response_format: Desired response format (e.g. "url" or "b64_json" for
            image requests). None means use the default for the request type.
    """

    prompt: Union[OmniTextPrompt, Dict[str, Any]]
    sampling_params_list: list | None = None
    request_type: RequestType = RequestType.CHAT_COMPLETION
    fps: int = 0
    response_format: str | None = None


class OmniHandler(BaseOmniHandler):
    """Unified handler for multi-stage pipelines using vLLM-Omni.

    Handles text-to-text, text-to-image, text-to-video, and text-to-audio generation.
    """

    def __init__(
        self,
        runtime,
        config,
        default_sampling_params: Dict[str, Any],
        shutdown_event: asyncio.Event | None = None,
        media_output_fs: Optional[DirFileSystem] = None,
        media_output_http_url: Optional[str] = None,
    ):
        """Initialize the unified Omni handler.

        Args:
            runtime: Dynamo distributed runtime.
            component: Dynamo component handle.
            config: Parsed Config object from args.py.
            default_sampling_params: Default sampling parameters dict.
            shutdown_event: Optional asyncio event for graceful shutdown.
            media_output_fs: Filesystem for storing generated images/videos.
            media_output_http_url: Base URL for rewriting media paths in responses.
        """
        super().__init__(
            runtime=runtime,
            config=config,
            default_sampling_params=default_sampling_params,
            shutdown_event=shutdown_event,
        )
        self.media_output_fs = media_output_fs
        self.media_output_http_url = media_output_http_url

        # Cache TTS capabilities from model config (once at init, reused per request).
        # Mirrors vLLM-Omni's OmniOpenAIServingSpeech.__init__().
        self._tts_supported_speakers: set = self._load_supported_speakers()
        self._tts_supported_languages: set = self._load_supported_languages()
        if self._tts_supported_speakers:
            logger.info(
                f"Loaded {len(self._tts_supported_speakers)} TTS speakers: "
                f"{sorted(self._tts_supported_speakers)}"
            )
        if self._tts_supported_languages:
            logger.info(
                f"Loaded {len(self._tts_supported_languages)} TTS languages: "
                f"{sorted(self._tts_supported_languages)}"
            )

    async def generate(
        self, request: Dict[str, Any], context
    ) -> AsyncGenerator[Dict, None]:
        """Generate outputs via the unified OpenAI mode.

        Args:
            request: Raw request dictionary from the Rust frontend.
            context: Dynamo context for request tracking.

        Yields:
            Response dictionaries.
        """
        request_id = context.id()
        logger.debug(f"Omni Request ID: {request_id}")

        async for chunk in self._generate_openai_mode(request, context, request_id):
            yield chunk

    async def _generate_openai_mode(
        self, request: Dict[str, Any], context, request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Single generation path for all request protocols and output modalities."""

        parsed_request, request_type = parse_request_type(
            request, self.config.output_modalities
        )
        inputs = await self.build_engine_inputs(parsed_request, request_type)

        generate_kwargs: Dict[str, Any] = {
            "prompt": inputs.prompt,
            "request_id": request_id,
        }
        if inputs.sampling_params_list is not None:
            generate_kwargs["sampling_params_list"] = inputs.sampling_params_list

        previous_text = ""

        async with self._abort_monitor(context, request_id):
            try:
                async for stage_output in self.engine_client.generate(
                    **generate_kwargs,
                ):
                    if (
                        stage_output.final_output_type == "text"
                        and stage_output.request_output
                    ):
                        chunk = self._format_text_chunk(
                            stage_output.request_output,
                            request_id,
                            previous_text,
                        )
                        if chunk:
                            output = stage_output.request_output.outputs[0]
                            previous_text = output.text
                            yield chunk

                    elif (
                        stage_output.final_output_type == "image"
                        and stage_output.images
                    ):
                        # vllm-omni uses final_output_type="image" for both
                        # image and video diffusion outputs. Use the parsed
                        # request type to route to the correct formatter.
                        if inputs.request_type == RequestType.VIDEO_GENERATION:
                            chunk = await self._format_video_chunk(
                                stage_output.images,
                                request_id,
                                fps=inputs.fps,
                            )
                        else:
                            chunk = await self._format_image_chunk(
                                stage_output.images,
                                request_id,
                                response_format=inputs.response_format,
                                request_type=inputs.request_type,
                            )
                        if chunk:
                            yield chunk

                    elif stage_output.final_output_type == "audio":
                        mm_output = stage_output.multimodal_output
                        if mm_output:
                            chunk = await self._format_audio_chunk(
                                mm_output,
                                request_id,
                                response_format=inputs.response_format,
                                request_type=inputs.request_type,
                            )
                            if chunk:
                                yield chunk

            except GeneratorExit:
                logger.info(f"Request {request_id} aborted due to shutdown")
                raise
            except Exception as e:
                logger.error(f"Error during generation for request {request_id}: {e}")
                yield self._error_chunk(request_id, str(e))

    async def build_engine_inputs(
        self,
        parsed_request: Union[
            NvCreateImageRequest,
            NvCreateVideoRequest,
            NvCreateAudioSpeechRequest,
            Dict[str, Any],
        ],
        request_type: RequestType,
    ) -> EngineInputs:
        """Convert a parsed request into AsyncOmni engine inputs.

        Args:
            parsed_request: Output from parse_request_type -- a Pydantic model
                for image/video/audio requests, or a raw dict for chat completions.
            request_type: The RequestType determined by parse_request_type.

        Returns:
            EngineInputs ready for engine_client.generate().
        """
        if request_type == RequestType.CHAT_COMPLETION:
            return self._engine_inputs_from_chat(parsed_request)
        elif request_type == RequestType.IMAGE_GENERATION:
            return self._engine_inputs_from_image(parsed_request)
        elif request_type == RequestType.VIDEO_GENERATION:
            return self._engine_inputs_from_video(parsed_request)
        elif request_type == RequestType.AUDIO_GENERATION:
            return await self._engine_inputs_from_audio(parsed_request)

        raise ValueError(f"Unknown request type: {request_type}")

    def _engine_inputs_from_chat(self, request: Dict[str, Any]) -> EngineInputs:
        """Build engine inputs from a chat completions request dict."""

        text_prompt = self._extract_text_prompt(request)
        if text_prompt is None:
            raise ValueError("No user message found in chat completion request")

        prompt = OmniTextPrompt(prompt=text_prompt)

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=None,
            request_type=RequestType.CHAT_COMPLETION,
            fps=0,
        )

    def _engine_inputs_from_image(self, req: NvCreateImageRequest) -> EngineInputs:
        """Build engine inputs from an NvCreateImageRequest."""
        width, height = parse_size(req.size, default_w=1024, default_h=1024)
        nvext = req.nvext

        prompt = OmniTextPrompt(
            prompt=req.prompt,
            negative_prompt=nvext.negative_prompt
            if nvext and nvext.negative_prompt
            else None,
        )

        sp = OmniDiffusionSamplingParams(
            height=height,
            width=width,
        )
        if req.n is not None:
            sp.num_outputs_per_prompt = req.n
        if nvext:
            if nvext.num_inference_steps is not None:
                sp.num_inference_steps = nvext.num_inference_steps
            if nvext.guidance_scale is not None:
                sp.guidance_scale = nvext.guidance_scale
            if nvext.seed is not None:
                sp.seed = nvext.seed

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=[sp],
            request_type=RequestType.IMAGE_GENERATION,
            response_format=req.response_format,
        )

    def _engine_inputs_from_video(self, req: NvCreateVideoRequest) -> EngineInputs:
        """Build engine inputs from an NvCreateVideoRequest."""
        width, height = parse_size(req.size)
        nvext = req.nvext

        nvext_fps = nvext.fps if nvext else None
        nvext_num_frames = nvext.num_frames if nvext else None

        num_frames = compute_num_frames(
            num_frames=nvext_num_frames,
            seconds=req.seconds,
            fps=nvext_fps,
            default_fps=DEFAULT_VIDEO_FPS,
        )
        fps = nvext_fps if nvext_fps is not None else DEFAULT_VIDEO_FPS

        prompt = OmniTextPrompt(
            prompt=req.prompt,
            negative_prompt=nvext.negative_prompt
            if nvext and nvext.negative_prompt
            else None,
        )

        sp = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=num_frames,
        )
        if nvext:
            if nvext.num_inference_steps is not None:
                sp.num_inference_steps = nvext.num_inference_steps
            if nvext.guidance_scale is not None:
                sp.guidance_scale = nvext.guidance_scale
            if nvext.seed is not None:
                sp.seed = nvext.seed
        if fps is not None:
            sp.fps = fps

        logger.info(
            f"Video diffusion request: prompt='{req.prompt[:50]}...', "
            f"size={width}x{height}, frames={num_frames}, fps={fps}"
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=[sp],
            request_type=RequestType.VIDEO_GENERATION,
            fps=fps,
        )

    # -- TTS capability loading from model config -----------------------------

    def _load_supported_speakers(self) -> set:
        """Load supported speakers from model config (case-insensitive).

        Reads ``hf_config.talker_config.spk_id`` or ``speaker_id``,
        matching vLLM-Omni's ``_load_supported_speakers()``.
        """
        try:
            hf_config = self.engine_client.model_config.hf_config
            talker_config = getattr(hf_config, "talker_config", None)
            if talker_config is None:
                return set()
            for attr_name in ("spk_id", "speaker_id"):
                speakers_dict = getattr(talker_config, attr_name, None)
                if speakers_dict and isinstance(speakers_dict, dict):
                    return {s.lower() for s in speakers_dict.keys()}
        except Exception as e:
            logger.warning("Could not load speakers from model config: %s", e)
        return set()

    def _load_supported_languages(self) -> set:
        """Load supported languages from model config.

        Reads ``hf_config.talker_config.codec_language_id``.
        """
        try:
            hf_config = self.engine_client.model_config.hf_config
            talker_config = getattr(hf_config, "talker_config", None)
            if talker_config is None:
                return set()
            lang_dict = getattr(talker_config, "codec_language_id", None)
            if lang_dict and isinstance(lang_dict, dict):
                return {lang.lower() for lang in lang_dict.keys()}
        except Exception as e:
            logger.warning("Could not load languages from model config: %s", e)
        return set()

    # -- TTS model detection --------------------------------------------------

    def _is_tts_model(self) -> bool:
        """Check if the loaded model is a Qwen3-TTS-style model.

        Mirrors vLLM-Omni's _find_tts_stage(): iterates over the AsyncOmni
        stage_list and returns True if any stage's ``model_stage`` is in
        ``_TTS_MODEL_STAGES``.  Non-TTS audio models (MiMo-Audio,
        Qwen3-Omni, Stable Audio, …) will return False and use a plain
        text prompt instead.
        """
        stage_list = getattr(self.engine_client, "stage_list", None)
        if stage_list is None:
            return False
        return any(
            getattr(stage, "model_stage", None) in _TTS_MODEL_STAGES
            for stage in stage_list
        )

    # -- Audio engine input construction --------------------------------------

    async def _engine_inputs_from_audio(
        self, req: NvCreateAudioSpeechRequest
    ) -> EngineInputs:
        """Build engine inputs for an audio/TTS request.

        Two code paths (matching vLLM-Omni serving_speech.py):

        * **TTS path** (Qwen3-TTS): builds ``prompt_token_ids`` +
          ``additional_information`` with speaker / language / task_type
          parameters.  Includes validation, ref_audio resolution, and
          tokenizer-based prompt length estimation.

        * **Generic audio path** (MiMo-Audio, Qwen3-Omni, Stable Audio,
          …): sends a plain ``{"prompt": text}`` and lets the model handle
          the rest – identical to image / video diffusion prompts.
        """
        if not req.input or not req.input.strip():
            raise ValueError("Input text cannot be empty")

        if self._is_tts_model():
            return await self._engine_inputs_tts(req)

        # Generic audio model – plain text prompt (same as image/video)
        prompt = OmniTextPrompt(prompt=req.input)
        logger.info(
            f"Audio request (generic): input='{req.input[:50]}...'"
        )
        return EngineInputs(
            prompt=prompt,
            sampling_params_list=None,
            request_type=RequestType.AUDIO_GENERATION,
            response_format=req.response_format,
        )

    # -- Qwen3-TTS-specific helpers -------------------------------------------

    async def _engine_inputs_tts(
        self, req: NvCreateAudioSpeechRequest
    ) -> EngineInputs:
        """Build engine inputs for Qwen3-TTS models.

        Constructs the ``prompt_token_ids`` + ``additional_information``
        prompt format expected by ``Qwen3TTSForConditionalGeneration``.
        """
        self._validate_tts_request(req)

        # Normalize voice to lowercase (case-insensitive matching)
        if req.voice is not None:
            req.voice = req.voice.lower()

        task_type = req.task_type or "CustomVoice"

        # Build TTS parameters following vLLM-Omni's _build_tts_params()
        tts_params: Dict[str, Any] = {
            "text": [req.input],
            "task_type": [task_type],
            "language": [req.language or "Auto"],
            "instruct": [req.instructions or ""],
            "max_new_tokens": [req.max_new_tokens or 2048],
        }

        # Speaker — default to Vivian for CustomVoice (matching vLLM-Omni)
        if req.voice is not None:
            tts_params["speaker"] = [req.voice]
        elif task_type == "CustomVoice":
            tts_params["speaker"] = ["Vivian"]

        # Voice cloning params (Base task)
        if req.ref_audio is not None:
            wav_list, sr = await self._resolve_ref_audio(req.ref_audio)
            tts_params["ref_audio"] = [[wav_list, sr]]
        if req.ref_text is not None:
            tts_params["ref_text"] = [req.ref_text]

        # VoiceDesign requires non_streaming_mode
        if task_type == "VoiceDesign":
            tts_params["non_streaming_mode"] = [True]

        # Estimate prompt length using tokenizer (fallback: 2048)
        estimated_len = self._estimate_tts_prompt_len(tts_params)

        prompt = {
            "prompt_token_ids": [1] * estimated_len,
            "additional_information": tts_params,
        }

        logger.info(
            f"Audio TTS request: input='{req.input[:50]}...', "
            f"voice={tts_params.get('speaker', ['N/A'])[0]}, "
            f"task_type={task_type}, prompt_len={estimated_len}"
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=None,
            request_type=RequestType.AUDIO_GENERATION,
            response_format=req.response_format,
        )

    def _validate_tts_request(self, req: NvCreateAudioSpeechRequest) -> None:
        """Validate Qwen3-TTS-specific request parameters.

        Uses dynamically loaded speakers and languages from the model's
        ``config.json`` (``talker_config.spk_id`` and
        ``talker_config.codec_language_id``).  Falls back to the hardcoded
        ``_TTS_LANGUAGES`` set when the model config is unavailable.

        Raises ValueError on invalid input.  Only called on the TTS code
        path – generic audio models skip this.
        """
        task_type = req.task_type or "CustomVoice"

        # Validate language against model config (dynamic) or fallback set
        if req.language is not None:
            supported_langs = self._tts_supported_languages or {
                lang.lower() for lang in _TTS_LANGUAGES
            }
            # Model config uses lowercase keys ("english"), but the API
            # accepts title-case ("English") and "Auto".
            if req.language.lower() not in supported_langs and req.language != "Auto":
                raise ValueError(
                    f"Invalid language '{req.language}'. "
                    f"Supported: Auto, {', '.join(sorted(supported_langs))}"
                )

        # Validate speaker against model config (dynamic)
        if task_type == "CustomVoice" and req.voice is not None:
            if self._tts_supported_speakers:
                if req.voice.lower() not in self._tts_supported_speakers:
                    raise ValueError(
                        f"Invalid voice '{req.voice}'. "
                        f"Supported: {', '.join(self._tts_supported_speakers)}"
                    )

        if task_type == "Base" and req.ref_audio is None:
            raise ValueError("Base task requires 'ref_audio' for voice cloning")

        if task_type != "Base":
            if req.ref_text is not None:
                raise ValueError("'ref_text' is only valid for Base task")

        if task_type == "VoiceDesign" and not req.instructions:
            raise ValueError(
                "VoiceDesign task requires 'instructions' to describe the voice"
            )

        if (
            req.instructions
            and len(req.instructions) > _TTS_MAX_INSTRUCTIONS_LENGTH
        ):
            raise ValueError(
                f"Instructions too long (max {_TTS_MAX_INSTRUCTIONS_LENGTH} characters)"
            )

        if req.max_new_tokens is not None:
            if req.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                raise ValueError(
                    f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
                )
            if req.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                raise ValueError(
                    f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"
                )

    async def _resolve_ref_audio(self, ref_audio_str: str) -> tuple:
        """Download or decode reference audio for voice cloning (Base task).

        Supports HTTP/HTTPS URLs and base64 data URIs.
        Returns ``(wav_samples_list, sample_rate)``.
        """
        import io

        import soundfile as sf

        if ref_audio_str.startswith(("http://", "https://")):
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    ref_audio_str,
                    timeout=aiohttp.ClientTimeout(total=_REF_AUDIO_TIMEOUT_S),
                ) as resp:
                    if resp.status != 200:
                        raise ValueError(
                            f"Failed to download ref_audio: HTTP {resp.status}"
                        )
                    audio_bytes = await resp.read()
                    if len(audio_bytes) > _REF_AUDIO_MAX_BYTES:
                        raise ValueError(
                            f"ref_audio too large "
                            f"({len(audio_bytes)} bytes, max {_REF_AUDIO_MAX_BYTES})"
                        )
        elif ref_audio_str.startswith("data:"):
            _, encoded = ref_audio_str.split(",", 1)
            audio_bytes = base64.b64decode(encoded)
        else:
            raise ValueError(
                "ref_audio must be a URL (http/https) or base64 data URI (data:...)"
            )

        wav_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        return wav_data, int(sr)

    def _estimate_tts_prompt_len(self, tts_params: Dict[str, Any]) -> int:
        """Estimate Qwen3-TTS prompt length using its tokenizer.

        Falls back to 2048 if the model-specific estimator is unavailable
        (e.g. when a future TTS model is added to ``_TTS_MODEL_STAGES``
        but doesn't expose the same static method).
        """
        try:
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
                Qwen3TTSTalkerForConditionalGeneration,
            )

            if not hasattr(self, "_tts_tokenizer") or self._tts_tokenizer is None:
                from transformers import AutoTokenizer

                self._tts_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model,
                    trust_remote_code=True,
                    padding_side="left",
                )

            hf_config = self.engine_client.model_config.hf_config
            talker_config = getattr(hf_config, "talker_config", None)
            task_type = (tts_params.get("task_type") or ["CustomVoice"])[0]

            return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
                additional_information=tts_params,
                task_type=task_type,
                tokenize_prompt=lambda t: self._tts_tokenizer(t, padding=False)[
                    "input_ids"
                ],
                codec_language_id=getattr(talker_config, "codec_language_id", None)
                if talker_config
                else None,
                spk_is_dialect=getattr(talker_config, "spk_is_dialect", None)
                if talker_config
                else None,
            )
        except Exception as e:
            logger.warning(
                "Failed to estimate TTS prompt length, using fallback 2048: %s", e
            )
            return 2048

    def _extract_audio_tensor(
        self, mm_output: Dict[str, Any]
    ) -> tuple:
        """Extract audio tensor and sample rate from multimodal_output dict.

        vLLM-Omni TTS models return audio in multimodal_output with keys
        "audio" or "model_outputs" for the waveform, and "sr" for sample rate.

        Returns:
            (audio_numpy, sample_rate) tuple.
        """
        import numpy as np
        import torch

        # Find audio tensor — key is "audio" or "model_outputs"
        audio_key = "audio" if "audio" in mm_output else "model_outputs"
        audio_val = mm_output.get(audio_key)
        if audio_val is None:
            raise ValueError(
                f"No audio data in multimodal_output. Keys: {list(mm_output.keys())}"
            )

        # Handle cumulative list mode (streaming chunks)
        if isinstance(audio_val, list):
            audio_val = torch.cat(audio_val, dim=-1)

        # Convert to numpy float32
        if hasattr(audio_val, "float"):
            audio_np = audio_val.float().detach().cpu().numpy()
        elif isinstance(audio_val, np.ndarray):
            audio_np = audio_val.astype(np.float32)
        else:
            audio_np = np.array(audio_val, dtype=np.float32)

        # Squeeze extra dimensions (e.g. [1, N] -> [N])
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        # Extract sample rate
        sr_raw = mm_output.get("sr", 24000)
        if isinstance(sr_raw, list):
            sr_raw = sr_raw[-1] if sr_raw else 24000
        sample_rate = sr_raw.item() if hasattr(sr_raw, "item") else int(sr_raw)

        return audio_np, sample_rate

    def _encode_audio(
        self, audio_np, sample_rate: int, fmt: str = "wav", speed: float = 1.0
    ) -> tuple:
        """Encode a numpy float32 waveform to audio bytes.

        Uses soundfile for multi-format support (wav, pcm, flac, mp3, aac, opus).
        Applies speed adjustment via librosa if speed != 1.0.

        Returns:
            (audio_bytes, media_type) tuple.
        """
        import soundfile as sf

        # Apply speed adjustment
        if speed != 1.0:
            try:
                import librosa

                audio_np = librosa.effects.time_stretch(y=audio_np, rate=speed)
            except ImportError:
                logger.warning(
                    "librosa not installed, ignoring speed adjustment"
                )

        fmt = (fmt or "wav").lower()
        format_map = {
            "wav": ("WAV", "audio/wav", {}),
            "pcm": ("RAW", "audio/pcm", {"subtype": "PCM_16"}),
            "flac": ("FLAC", "audio/flac", {}),
            "mp3": ("MP3", "audio/mpeg", {}),
            "aac": ("AAC", "audio/aac", {}),
            "opus": ("OGG", "audio/ogg", {"subtype": "OPUS"}),
        }

        if fmt not in format_map:
            logger.warning(f"Unsupported format '{fmt}', defaulting to wav")
            fmt = "wav"

        sf_format, media_type, kwargs = format_map[fmt]

        buf = BytesIO()
        sf.write(buf, audio_np, sample_rate, format=sf_format, **kwargs)
        return buf.getvalue(), media_type

    async def _format_audio_chunk(
        self,
        mm_output: Dict[str, Any],
        request_id: str,
        response_format: str | None = None,
        request_type: RequestType = RequestType.AUDIO_GENERATION,
        speed: float = 1.0,
    ) -> Dict[str, Any] | None:
        """Format multimodal audio output for the response.

        Args:
            mm_output: The multimodal_output dict from OmniRequestOutput,
                containing audio tensors ("audio" or "model_outputs") and
                sample rate ("sr").
            request_id: Unique request identifier.
            response_format: Audio format (wav, mp3, pcm, etc.) or "url".
            request_type: Chat completion or dedicated audio generation.
            speed: Speed adjustment factor (1.0 = normal).

        Returns:
            Formatted response dict, or None if no audio.
        """
        if not mm_output:
            return self._error_chunk(request_id, "No audio generated")

        try:
            start_time = time.time()

            # Extract tensor
            audio_np, sample_rate = self._extract_audio_tensor(mm_output)

            # Determine encoding format (url is a storage mode, not an audio format)
            encode_fmt = "wav" if response_format in (None, "url", "b64_json") else response_format

            # Encode audio with format and speed
            audio_bytes, media_type = await asyncio.to_thread(
                self._encode_audio, audio_np, sample_rate, encode_fmt, speed
            )

            logger.info(
                f"Audio encoded for request {request_id}: "
                f"{len(audio_np)} samples, sr={sample_rate}, "
                f"{len(audio_bytes)} bytes {encode_fmt}"
            )

            inference_time = time.time() - start_time

            # Build response data
            if response_format == "url":
                ext = encode_fmt if encode_fmt != "opus" else "ogg"
                storage_path = f"audios/{request_id}/{uuid.uuid4()}.{ext}"
                url = await upload_to_fs(
                    self.media_output_fs,
                    storage_path,
                    audio_bytes,
                    self.media_output_http_url,
                )
                audio_data_obj = AudioData(url=url)
            else:
                b64 = base64.b64encode(audio_bytes).decode("utf-8")
                audio_data_obj = AudioData(b64_json=b64)

            response = NvAudioSpeechResponse(
                id=request_id,
                object="audio.speech",
                model=self.config.served_model_name or self.config.model,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[audio_data_obj],
                inference_time_s=inference_time,
            )
            return response.model_dump()

        except Exception as e:
            logger.error(f"Failed to process audio for request {request_id}: {e}")
            error_response = NvAudioSpeechResponse(
                id=request_id,
                object="audio.speech",
                model=self.config.served_model_name or self.config.model,
                status="failed",
                progress=0,
                created=int(time.time()),
                data=[],
                error=str(e),
            )
            return error_response.model_dump()

    async def _prepare_image_output(
        self, images: list, request_id: str, response_format: str | None = None
    ) -> list:
        """Prepare image output for response.

        Args:
            images: List of PIL Image objects.
            request_id: Unique request identifier.
            response_format: Response format ("url" or "b64_json").

        Returns:
            List of image URLs or base64 data-URL strings.
        """
        outlist = []

        for img in images:
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            if response_format == "url":
                storage_path = f"images/{request_id}/{uuid.uuid4()}.png"
                url = await upload_to_fs(
                    self.media_output_fs,
                    storage_path,
                    image_bytes,
                    self.media_output_http_url,
                )
                outlist.append(url)
            elif response_format == "b64_json" or response_format is None:
                img_base64 = base64.b64encode(image_bytes).decode("utf-8")
                data_url = f"data:image/png;base64,{img_base64}"
                outlist.append(data_url)
            else:
                raise ValueError(f"Invalid response format: {response_format}")
        return outlist

    async def _format_image_chunk(
        self,
        images: list,
        request_id: str,
        response_format: str | None = None,
        request_type: RequestType = RequestType.IMAGE_GENERATION,
    ) -> Dict[str, Any] | None:
        """Format image output for the appropriate endpoint response.

        Args:
            images: List of PIL Image objects generated by AsyncOmni engine.
            request_id: Unique request identifier.
            response_format: Response format (url, b64_json, None).
            request_type: Request type (chat completion, image generation).

        Returns:
            Formatted response dict, or None if no images generated.
        """
        if not images:
            return self._error_chunk(request_id, "No images generated")

        data_urls = await self._prepare_image_output(
            images, request_id, response_format
        )

        if request_type == RequestType.CHAT_COMPLETION:
            chunk = {
                "id": request_id,
                "created": int(time.time()),
                "object": "chat.completion.chunk",
                "model": self.config.served_model_name or self.config.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url}}
                                for data_url in data_urls
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            return chunk
        elif request_type == RequestType.IMAGE_GENERATION:
            image_data_list = []
            for data_url in data_urls:
                if response_format == "url":
                    image_data_list.append(ImageData(url=data_url))
                elif response_format == "b64_json" or response_format is None:
                    if data_url.startswith("data:image"):
                        _, b64_part = data_url.split(",", 1)
                        image_data_list.append(ImageData(b64_json=b64_part))
                    else:
                        image_data_list.append(ImageData(b64_json=data_url))
                else:
                    raise ValueError(f"Invalid response format: {response_format}")

            output = NvImagesResponse(created=int(time.time()), data=image_data_list)
            return output.model_dump()

    async def _format_video_chunk(
        self,
        images: list,
        request_id: str,
        fps: int,
    ) -> Dict[str, Any] | None:
        """Convert diffusion output frames to MP4 and return as NvVideosResponse.

        Args:
            images: List of PIL Image frames from the diffusion stage.
            request_id: Unique request identifier.
            fps: Frames per second for the output video.

        Returns:
            ``NvVideosResponse.model_dump()`` dict, or ``None`` if no frames.
        """
        if not images:
            return None

        try:
            start_time = time.time()

            frame_list = normalize_video_frames(images)

            logger.info(
                f"Encoding {len(frame_list)} frames to MP4 for request {request_id} "
                f"(fps={fps})"
            )

            # Encode frames to MP4 via temp file, then read bytes for upload
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                await asyncio.to_thread(export_to_video, frame_list, tmp.name, fps)
                video_bytes = tmp.read()

            # Upload via filesystem
            storage_path = f"videos/{request_id}.mp4"
            video_url = await upload_to_fs(
                self.media_output_fs,
                storage_path,
                video_bytes,
                self.media_output_http_url,
            )

            logger.info(f"Video uploaded to {video_url} for request {request_id}")

            inference_time = time.time() - start_time

            response = NvVideosResponse(
                id=request_id,
                object="video",
                model=self.config.served_model_name or self.config.model,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[VideoData(url=video_url)],
                inference_time_s=inference_time,
            )
            return response.model_dump()

        except Exception as e:
            logger.error(f"Failed to encode video for request {request_id}: {e}")
            error_response = NvVideosResponse(
                id=request_id,
                object="video",
                model=self.config.served_model_name or self.config.model,
                status="failed",
                progress=0,
                created=int(time.time()),
                data=[],
                error=str(e),
            )
            return error_response.model_dump()

    def _format_text_chunk(
        self,
        request_output,
        request_id: str,
        previous_text: str,
    ) -> Dict[str, Any] | None:
        """Format text output as OpenAI chat completion chunk."""
        if not request_output.outputs:
            return self._error_chunk(request_id, "No outputs from engine")

        output = request_output.outputs[0]

        # Calculate delta text (new text since last chunk)
        delta_text = output.text[len(previous_text) :]

        chunk = {
            "id": request_id,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "model": self.config.served_model_name or self.config.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": delta_text,
                    },
                    "finish_reason": self._normalize_finish_reason(output.finish_reason)
                    if output.finish_reason
                    else None,
                }
            ],
        }

        # Add usage on final chunk
        if output.finish_reason:
            chunk["usage"] = self._build_completion_usage(request_output)

        return chunk
