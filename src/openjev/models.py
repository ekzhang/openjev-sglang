from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .config import MAX_ANSWERS, MAX_QUESTIONS

Content = str | dict[str, JsonValue] | list[JsonValue]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TextPart(StrictModel):
    type: Literal["text"]
    text: str


class ImageUrl(StrictModel):
    url: str = Field(min_length=1)


class ImagePart(StrictModel):
    """An image inside a chat message's content list.

    Only the OpenAI-style `{"type": "image_url", "image_url": {"url": ...}}` shape is
    accepted. Qwen's chat template matches on the literal string `image_url`, so that
    key name must stay exactly this. The URL may be an `http(s)://` URL or a
    `data:image/...;base64,...` payload; SGLang loads either form itself.
    """

    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


def image_urls(content: Any) -> list[str]:
    """Every image URL inside a message content list, in model-visible order."""
    if not isinstance(content, list):
        return []
    return [
        part["image_url"]["url"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "image_url"
        and isinstance(part.get("image_url"), dict)
        and isinstance(part["image_url"].get("url"), str)
        and part["image_url"]["url"]
    ]


class NoulCriteria(StrictModel):
    yes: str = Field(default="Yes", alias="true", description="Meaning of a positive answer.")
    no: str = Field(default="No", alias="false", description="Meaning of a negative answer.")


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Content | None = Field(
        default=None, description="The yes/no question or evaluation instructions."
    )
    criteria: NoulCriteria = Field(
        default_factory=NoulCriteria, description="Optional definitions of true and false."
    )


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Content | None = Field(
        default=None, description="Question or instructions for choosing one option."
    )
    criteria: dict[str, str | None] = Field(
        min_length=2,
        max_length=MAX_ANSWERS,
        description=(
            "Map of 2–64 option keys to descriptions. The model sees only the description, "
            "or the option key when its description is null. Returned answers use the keys."
        ),
    )


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Content | None = Field(
        default=None, description="Question or instructions for applying the rubric."
    )
    criteria: list[str] = Field(
        min_length=2,
        max_length=MAX_ANSWERS,
        description=(
            "Ordered rubric, lowest to highest. Scores use zero-based indices: three "
            "descriptions correspond to levels 0, 1, and 2."
        ),
    )


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "jev-latest",
                "state": "I was charged twice. Please refund the duplicate.",
                "questions": {
                    "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
                    "department": {
                        "type": "choice",
                        "instructions": "Which department should handle this?",
                        "criteria": {
                            "billing": "Payments and refunds",
                            "technical": "Software bugs",
                        },
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgent is this?",
                        "criteria": ["Routine", "Urgent", "Emergency"],
                    },
                },
            }
        }
    )
    state: Content = Field(
        description="Shared text, structured JSON, or a text chat transcript to evaluate."
    )
    model: str = Field(
        min_length=1,
        description=(
            "Model name from GET /v1/models. jev-latest selects this deployment's configured model."
        ),
    )
    questions: dict[str, Question] = Field(
        min_length=1,
        max_length=MAX_QUESTIONS,
        description=(
            "1–64 questions keyed by your own identifiers. The response preserves these keys."
        ),
    )


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(
        ge=0, le=1, description="Probability of true, normalized over the true and false options."
    )


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str = Field(description="Option key with the highest probability.")
    probabilities: dict[str, float] = Field(
        description="Probability for every supplied option; values sum to 1."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "Distribution concentration: 1 minus normalized entropy. Not calibrated correctness."
        ),
    )


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float = Field(
        ge=0,
        description="Expected zero-based rubric index, between 0 and number of levels minus 1.",
    )
    legend: dict[str, str] = Field(
        description="Stringified zero-based index mapped to each rubric description."
    )
    probabilities: dict[str, float] = Field(
        description="Probability for each rubric index; values sum to 1."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "Distribution concentration: 1 minus normalized entropy. Not calibrated correctness."
        ),
    )


class Usage(StrictModel):
    input_tokens: int = Field(
        ge=0,
        description="Total submitted prompt tokens, including cached tokens and shared warm-up.",
    )
    output_tokens: int = Field(
        ge=0, description="One warm-up token plus one token per question; N + 1 for N questions."
    )


class SystemOneResponse(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "jev-latest",
                "answers": {
                    "refund": {"type": "noul", "noul": 0.99},
                    "department": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.99, "technical": 0.01},
                        "confidence": 0.9192,
                    },
                    "urgency": {
                        "type": "score",
                        "score": 0.4,
                        "legend": {"0": "Routine", "1": "Urgent", "2": "Emergency"},
                        "probabilities": {"0": 0.65, "1": 0.3, "2": 0.05},
                        "confidence": 0.2799,
                    },
                },
                "usage": {"input_tokens": 480, "output_tokens": 4},
            }
        }
    )
    model: str = Field(description="The model name or alias supplied in the request.")
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = Field(
        description="Typed answers keyed by the same identifiers as the request's questions."
    )
    usage: Usage


class HealthResponse(StrictModel):
    status: Literal["ok", "unavailable"] = Field(
        description="Whether the checked component is available."
    )
    startup_seconds: float | None = Field(
        default=None, description="Seconds spent initializing this container; null if unavailable."
    )


class LiveResponse(StrictModel):
    status: Literal["ok"] = "ok"


class ModelEntry(StrictModel):
    name: str = Field(description="Accepted model ID or compatibility alias.")
    description: str = Field(description="Checkpoint served by this deployment.")
    release_date: str = Field(
        description=(
            "OpenJev catalogue entry date (YYYY-MM-DD), not the model's original release date."
        )
    )


class OpenAIModelEntry(StrictModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "openjev"


class ModelsResponse(StrictModel):
    models: list[ModelEntry] = Field(
        description=(
            "TypeSafe-compatible model catalogue, including jev-latest and the checkpoint ID."
        )
    )
    object: Literal["list"] = "list"
    data: list[OpenAIModelEntry] = Field(
        description="The same accepted IDs in OpenAI-compatible list format."
    )


class LimitsResponse(StrictModel):
    max_answers_per_question: int = Field(
        description="Maximum number of Choice options or Score levels."
    )
    max_questions: int = Field(description="Maximum questions per evaluation.")
    max_body_bytes: int = Field(description="Maximum JSON request body size in bytes.")
    max_input_tokens: int = Field(
        description="Maximum tokens per question branch, including its one output token."
    )
    max_total_input_tokens: int = Field(
        description=(
            "Maximum submitted prompt tokens summed over warm-up and all branches, "
            "including cache hits."
        )
    )
    max_concurrent_requests: int = Field(
        description="Maximum active evaluations per API container. Additional requests return 529."
    )
    max_images: int = Field(
        description=(
            "Maximum images per request's state. 0 means this deployment cannot evaluate "
            "images; such requests return 422."
        )
    )


class ErrorDetail(StrictModel):
    message: str = Field(description="Human-readable explanation of the failure.")


class ErrorResponse(StrictModel):
    error: ErrorDetail

