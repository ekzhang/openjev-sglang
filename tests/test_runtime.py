import pytest

from openjev.config import Settings
from openjev.runtime import backend_command


def test_explicit_model_and_revision_override_profile():
    settings = Settings(model="local/custom", revision="revision")
    assert settings.model == "local/custom"
    assert settings.model_revision == "revision"


def test_default_launch_keeps_radix_and_breakable_graphs():
    command = backend_command(Settings(), [])
    assert command[command.index("--cuda-graph-backend-prefill") + 1] == "breakable"
    assert command[command.index("--mamba-radix-cache-strategy") + 1] == "extra_buffer"
    assert "--disable-radix-cache" not in command
    assert "--speculative-algorithm" not in command
    assert "--revision" in command


def test_public_name_is_independent_of_weight_source(monkeypatch):
    monkeypatch.setenv("OPENJEV_SERVED_MODEL_NAME", "my-model")
    settings = Settings()
    command = backend_command(settings, [])
    assert command[command.index("--model-path") + 1] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert command[command.index("--served-model-name") + 1] == "my-model"


def test_reject_cache_disable():
    with pytest.raises(ValueError, match="radix"):
        backend_command(Settings(), ["--disable-radix-cache"])


def test_local_tokenizer_path_is_forwarded_to_rust():
    command = backend_command(Settings(), [], "/cache/pinned-snapshot")
    assert command[command.index("--tokenizer-path") + 1] == "/cache/pinned-snapshot"
    assert "--cuda-graph-max-bs-decode" in command
    assert "--cuda-graph-max-bs" not in command


def test_extra_flags_override_profile_defaults_without_orphan_values():
    settings = Settings(mem_fraction_static=0.9, moe_runner_backend="auto")
    command = backend_command(
        settings, ["--attention-backend", "flashinfer", "--kv-cache-dtype", "fp8_e4m3"]
    )
    # The profile defaults are pruned together with their values...
    assert "trtllm_mha" not in command
    assert "flashinfer_cutlass" not in command
    assert command.count("--attention-backend") == 1
    assert command[command.index("--attention-backend") + 1] == "flashinfer"
    assert command[command.index("--mem-fraction-static") + 1] == "0.9"
    assert command[command.index("--moe-runner-backend") + 1] == "auto"


def test_extra_flag_with_equals_does_not_swallow_next_arg():
    command = backend_command(Settings(), ["--attention-backend=flashinfer"])
    assert "--attention-backend=flashinfer" in command
    assert "trtllm_mha" not in command
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
