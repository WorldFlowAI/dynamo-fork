#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT


MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================="
echo "Starting vLLM-Omni Audio/TTS Worker"
echo "Model: $MODEL"
echo "=========================================="


echo "Starting frontend on port ${DYN_HTTP_PORT:-8000}..."
python -m dynamo.frontend &
FRONTEND_PID=$!

sleep 2

echo "Starting Omni Audio worker..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --omni \
    --output-modalities audio \
    --media-output-fs-url file:///tmp/dynamo_media \
    --trust-remote-code \
    --enforce-eager \
    "${EXTRA_ARGS[@]}"
