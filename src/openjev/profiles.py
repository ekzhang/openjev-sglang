"""Dependency-free deployment and backend presets; safe in Modal's outer Python."""

from dataclasses import dataclass, field

from .defaults import DEFAULT_MODEL, DEFAULT_REVISION, SGLANG_IMAGE


@dataclass(frozen=True)
class ModelProfile:
    model: str
    served_model_name: str
    revision: str
    image: str
    backend_python: str
    memory_mib: int
    app_name: str
    backend_args: tuple[str, ...]
    # Deployment-specific backend knobs. None falls back to the Settings default,
    # so an unset profile keeps the historical Modal/B200 command line exactly.
    moe_runner_backend: str | None = None
    mem_fraction_static: float | None = None
    attention_backend: str | None = None
    kv_cache_dtype: str | None = None
    mamba_radix_cache_strategy: str | None = None
    extra_backend_args: tuple[str, ...] = field(default=())


PROFILES = {
    "qwen36": ModelProfile(
        model=DEFAULT_MODEL,
        served_model_name="Qwen/Qwen3.6-35B-A3B",
        revision=DEFAULT_REVISION,
        image=SGLANG_IMAGE,
        backend_python="/opt/sglang/bin/python",
        memory_mib=32768,
        app_name="openjev-sglang",
        backend_args=(
            "--language-only",
            "--attention-backend",
            "trtllm_mha",
            "--kv-cache-dtype",
            "fp8_e4m3",
        ),
        mamba_radix_cache_strategy="extra_buffer",
    ),
    # Single RTX 4090 (Ada, SM89, 24 GB) on a host with a pip-installed SGLang,
    # instead of the B200 container the qwen36 profile targets. Point
    # OPENJEV_MODEL at a local AWQ weight directory.
    #
    # Why this differs from qwen36:
    #   * The default nvidia/Qwen3.6-35B-A3B-NVFP4 weights need Blackwell (SM100+)
    #     for NVFP4; Ada cannot execute them, so this profile uses AWQ 4-bit.
    #   * SGLang's AWQ MoE scheme asserts the MoE runner backend is "auto" and then
    #     selects awq_marlin itself; the qwen36 default aborts on that assert.
    #   * trtllm_mha attention only exists in NVIDIA's TensorRT-LLM container.
    #   * No --cpu-offload-gb: offloading puts an expert's gemma-style norm weight
    #     on the host while the loader adds it to a CUDA parameter, which aborts
    #     with "Expected all tensors to be on the same device ... cuda:0 and cpu!".
    #   * No --mamba-radix-cache-strategy: Qwen3-30B-A3B-Instruct-2507 is a plain
    #     Qwen3MoeForCausalLM with no linear-attention layers, so the hybrid-mamba
    #     cache does not apply. Passing the qwen36 default makes SGLang's
    #     _mamba_radix_cache_v2_req_prepare_for_extend index an unallocated
    #     mamba_ping_pong_track_buffer and abort with
    #     "TypeError: 'NoneType' object is not subscriptable" on the first prefill.
    "ada-4090-awq": ModelProfile(
        model="cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
        served_model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        revision="",
        image=SGLANG_IMAGE,
        backend_python="python3",
        memory_mib=32768,
        app_name="openjev-sglang-4090",
        backend_args=("--language-only",),
        moe_runner_backend="auto",
        mem_fraction_static=0.82,
        attention_backend="flashinfer",
        kv_cache_dtype="fp8_e4m3",
    ),
    # Same host, vision checkpoint: this is what makes image state work.
    #
    #   * No --language-only. That flag skips the vision tower, and SGLang rejects
    #     image input outright with "Multimodal inputs are not supported when
    #     --language-model-only is set; the encoder is not loaded."
    #   * Qwen3VLMoeForConditionalGeneration, AWQ 4-bit (group_size 128), so the
    #     30B/3B-active text tower plus the 27-layer vision tower fit a 24 GB card.
    #   * Lower static fraction than the text profile: the vision tower, the image
    #     embeddings and the much longer image-expanded prefixes all need room.
    "ada-4090-vl": ModelProfile(
        model="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        served_model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        revision="",
        image=SGLANG_IMAGE,
        backend_python="python3",
        memory_mib=32768,
        app_name="openjev-sglang-4090-vl",
        backend_args=(),
        moe_runner_backend="auto",
        mem_fraction_static=0.75,
        attention_backend="flashinfer",
        kv_cache_dtype="fp8_e4m3",
    ),
}
