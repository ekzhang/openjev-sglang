from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .defaults import DEFAULT_MODEL
from .defaults import MAX_ANSWERS as MAX_ANSWERS
from .defaults import MAX_QUESTIONS as MAX_QUESTIONS
from .profiles import PROFILES


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENJEV_", extra="ignore")

    profile: Literal["qwen36", "ada-4090-awq", "ada-4090-vl"] = "qwen36"
    model: str = DEFAULT_MODEL
    served_model_name: str = Field(default="Qwen/Qwen3.6-35B-A3B", min_length=1)
    revision: str | None = None
    model_alias: str = "jev-latest"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    backend_url: str = "http://127.0.0.1:30000"
    backend_port: int = Field(default=30000, ge=1, le=65535)
    sglang_python: str = "python3"
    frontend: Literal["rust", "python"] = "rust"
    # Backend tuning knobs. None means "use the selected profile's value"; the
    # qwen36 profile resolves to the historical B200 command line. Ada/host
    # deployments set these explicitly (AWQ needs a non-cutlass MoE runner, and a
    # 24 GB card needs a high static memory fraction).
    moe_runner_backend: str | None = None
    mem_fraction_static: float | None = Field(default=None, gt=0, le=1)
    attention_backend: str | None = None
    kv_cache_dtype: str | None = None
    api_key: SecretStr | None = None
    backend_api_key: SecretStr | None = None
    max_body_bytes: int = Field(default=2 * 1024 * 1024, gt=0)
    max_input_tokens: int = Field(default=32768, ge=256)
    max_total_input_tokens: int = Field(default=262144, gt=0)
    max_concurrent_requests: int = Field(default=16, gt=0)
    max_concurrent_branches: int = Field(default=64, gt=0)
    # Images attached to one request's state. 0 disables image state entirely, which
    # is the right default for the text-only deployments in PROFILES.
    max_images: int = Field(default=0, ge=0, le=16)
    request_timeout: float = Field(default=120, gt=0)
    startup_timeout: float = Field(default=1200, gt=0)
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def profile_defaults(cls, values):
        if isinstance(values, dict):
            profile = PROFILES.get(values.get("profile", "qwen36"))
            if profile is None:
                return values
            if "model" not in values:
                values = {**values, "model": profile.model}
            if "served_model_name" not in values:
                # The profile owns the public model id. `model` is only the weight
                # source and is routinely a local directory or a quantized
                # community build, so the public id must not be derived from it.
                values = {**values, "served_model_name": profile.served_model_name}
        return values

    @property
    def model_revision(self) -> str | None:
        if self.revision:
            return self.revision
        return next((p.revision for p in PROFILES.values() if p.model == self.model), None)
