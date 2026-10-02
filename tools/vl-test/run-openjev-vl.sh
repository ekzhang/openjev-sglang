#!/usr/bin/env bash
# OpenJev with a vision checkpoint (image state), for validating
# feat/multimodal-images on the single RTX 4090. Listens on 8010 so it can be
# swapped in while the text deployment on port 80 is stopped.
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/hf
export SGLANG_CACHE_DIR=/mnt/hf/runtime/sglang
export TRITON_CACHE_DIR=/mnt/hf/runtime/triton
export SGLANG_JIT_CACHE_DIR=/mnt/hf/runtime/jit
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
# flashinfer's JIT picks up the CUDA 13.1 headers vendored in the triton wheel
# while nvcc is 13.4; CCCL's guard aborts unless it is disabled.
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1"
export FLASHINFER_EXTRA_CFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1"
export OPENJEV_PROFILE=ada-4090-vl
export OPENJEV_MODEL=/mnt/hf/models/qwen3-vl-30b-awq
export OPENJEV_SERVED_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct
export OPENJEV_FRONTEND=rust
export OPENJEV_SGLANG_PYTHON=/usr/bin/python3
export OPENJEV_MAX_IMAGES=4
exec /opt/openjev/venv/bin/openjev serve --host 0.0.0.0 --port 8010
