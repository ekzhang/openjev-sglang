import copy

import pytest

from openjev.models import SystemOneRequest
from openjev.prompts import DEFAULT_INSTRUCTIONS, state_messages


def test_original_messages_are_preserved_and_questions_are_independent(compiler, payload):
    state = [
        {"role": "system", "content": "Original system instruction"},
        {"role": "user", "content": "Original user message"},
        {"role": "assistant", "content": "Original answer"},
        {"role": "user", "content": "One more question"},
    ]
    original = copy.deepcopy(state)
    payload["state"] = state
    request = SystemOneRequest.model_validate(payload)
    prepared = compiler.prepare(request)
    assert state == original
    assert state_messages(state) == (original, [])
    prefix = compiler.tokenizer.decode(prepared.prefix_ids)
    assert "Original system instruction" in prefix
    assert "One more question" in prefix
    for branch in prepared.branches:
        prompt = compiler.tokenizer.decode(branch.input_ids)
        assert prompt.startswith(prefix)
        assert prompt.count("Question:") == 1
        assert branch.question_id not in prompt.split("Question:")[1].split("Options:")[0]
        assert prompt.endswith("Answer:\n")


def test_structured_state_siblings_are_not_dropped():
    state = {"messages": [{"role": "user", "content": "Hi"}], "account": {"tier": "pro"}}
    messages, images = state_messages(state)
    assert '"tier":"pro"' in messages[0]["content"]
    assert images == []
    assert state_messages({"messages": state["messages"]}) == (state["messages"], [])


def test_question_ids_do_not_enter_the_model(compiler, payload):
    payload["questions"] = {"secret-id-never-in-prompt": payload["questions"]["yes"]}
    result = compiler.prepare(SystemOneRequest.model_validate(payload))
    assert "secret-id-never-in-prompt" not in compiler.tokenizer.decode(
        result.branches[0].input_ids
    )


def test_option_keys_do_not_enter_prompt_or_change_tokens(compiler, payload):
    payload["questions"] = {
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"secret-billing-key": "Payments", "secret-support-key": "Software\nBugs"},
        }
    }
    original = compiler.prepare(SystemOneRequest.model_validate(payload)).branches[0]
    prompt = compiler.tokenizer.decode(original.input_ids)
    assert "A: Payments\nB: Software\n   Bugs" in prompt
    assert "secret-" not in prompt
    assert '"option"' not in prompt
    assert '"description"' not in prompt
    payload["questions"]["team"]["criteria"] = {
        "other-key": "Payments",
        "renamed": "Software\nBugs",
    }
    renamed = compiler.prepare(SystemOneRequest.model_validate(payload)).branches[0]
    assert original.input_ids == renamed.input_ids
    assert original.option_keys == ["secret-billing-key", "secret-support-key"]
    assert renamed.option_keys == ["other-key", "renamed"]


def test_only_null_descriptions_fall_back_to_option_names(compiler, payload):
    payload["questions"] = {
        "q": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": {
                "Visible fallback": None,
                "Hidden key": "Visible description",
                "Also hidden": "",
            },
        }
    }
    branch = compiler.prepare(SystemOneRequest.model_validate(payload)).branches[0]
    prompt = compiler.tokenizer.decode(branch.input_ids)
    assert "A: Visible fallback\nB: Visible description\nC: " in prompt
    assert "Hidden key" not in prompt and "Also hidden" not in prompt


def test_missing_instructions_falls_back_to_default(compiler, payload):
    del payload["questions"]["yes"]["instructions"]
    branch = compiler.prepare(SystemOneRequest.model_validate(payload)).branches[0]
    prompt = compiler.tokenizer.decode(branch.input_ids)
    assert f"Question: {DEFAULT_INSTRUCTIONS}" in prompt


@pytest.mark.parametrize(
    "content",
    [
        12,
        [{"type": "image_url", "image_url": "http://x"}],  # old flat string form
        [{"type": "image_url", "image_url": {"url": ""}}],
        [{"type": "audio_url", "audio_url": {"url": "http://x"}}],
        [{"type": "text"}],
        ["not a part"],
    ],
)
def test_malformed_chat_content_is_rejected(content):
    with pytest.raises((ValueError, TypeError)):
        state_messages([{"role": "user", "content": content}])


def test_images_are_only_allowed_on_user_messages():
    with pytest.raises(ValueError, match="only supported in user messages"):
        state_messages(
            [{"role": "system", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}]
        )


def _image_state(*urls):
    return [
        {
            "role": "user",
            "content": [
                *(
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in urls
                ),
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]


def test_state_messages_returns_image_urls_in_order():
    messages, images = state_messages(_image_state("u1", "u2"))
    assert images == ["u1", "u2"]
    assert messages[0]["content"][-1]["text"] == "What is shown?"


def test_image_state_renders_one_placeholder_per_image(vision_compiler, payload):
    payload["state"] = _image_state("http://example.com/one.png", "data:image/png;base64,AAAA")
    prepared = vision_compiler.prepare(SystemOneRequest.model_validate(payload))
    assert prepared.image_data == [
        "http://example.com/one.png",
        "data:image/png;base64,AAAA",
    ]
    prefix = vision_compiler.tokenizer.decode(prepared.prefix_ids)
    # Exactly one placeholder per image: SGLang silently drops the image when a
    # placeholder is missing, and errors when it is pre-expanded.
    run = "<|vision_start|><|image_pad|><|vision_end|>"
    assert prefix.count(run) == 2
    assert prefix.count("<|image_pad|>") == 2
    # The URL itself must not be tokenized into the prompt.
    assert "example.com" not in prefix
    for branch in prepared.branches:
        assert prepared.prefix_ids == branch.input_ids[: len(prepared.prefix_ids)]


def test_text_only_state_sends_no_image_data(compiler, payload):
    prepared = compiler.prepare(SystemOneRequest.model_validate(payload))
    assert prepared.image_data == []


def test_images_rejected_when_model_has_no_vision_tokens(compiler, payload):
    # The default fixture tokenizer mirrors a text-only checkpoint.
    payload["state"] = _image_state("http://example.com/one.png")
    with pytest.raises(ValueError, match="cannot evaluate images"):
        compiler.prepare(SystemOneRequest.model_validate(payload))


def test_image_count_limit_is_enforced(vision_compiler, payload):
    payload["state"] = _image_state(*[f"http://example.com/{i}.png" for i in range(5)])
    with pytest.raises(ValueError, match="allows 4"):
        vision_compiler.prepare(SystemOneRequest.model_validate(payload))


def test_image_state_does_not_disturb_the_label_readout(vision_compiler, payload):
    payload["state"] = _image_state("http://example.com/one.png")
    prepared = vision_compiler.prepare(SystemOneRequest.model_validate(payload))
    for branch in prepared.branches:
        assert len(branch.label_ids) == len(branch.option_keys)
        assert all(isinstance(token_id, int) for token_id in branch.label_ids)


def test_all_answer_labels_are_distinct_single_tokens(compiler):
    assert len(compiler.labels) == 64
    assert len({token_id for _, token_id in compiler.labels}) == 64
    for text, token_id in compiler.labels:
        assert compiler.tokenizer.encode(text) == [token_id]
