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
    assert state_messages(state) == original
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
    assert '"tier":"pro"' in state_messages(state)[0]["content"]
    assert state_messages({"messages": state["messages"]}) == state["messages"]


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


def test_structured_criteria_are_serialized_for_every_question_type(compiler, payload):
    payload["questions"] = {
        "binary": {
            "type": "noul",
            "criteria": {
                "true": {"meaning": "Matches", "examples": ["alpha", "beta"]},
                "false": ["Different", {"reason": "No overlap"}],
            },
        },
        "choice": {
            "type": "choice",
            "criteria": {
                "hidden-object-key": {"meaning": "Object option"},
                "hidden-array-key": ["Array option", {"rank": 2}],
            },
        },
        "score": {
            "type": "score",
            "criteria": [{"level": "Low"}, ["High", {"threshold": 10}]],
        },
    }

    branches = {
        branch.question_id: compiler.tokenizer.decode(branch.input_ids)
        for branch in compiler.prepare(SystemOneRequest.model_validate(payload)).branches
    }

    assert 'A: {"meaning":"Matches","examples":["alpha","beta"]}' in branches["binary"]
    assert 'B: ["Different",{"reason":"No overlap"}]' in branches["binary"]
    assert 'A: {"meaning":"Object option"}' in branches["choice"]
    assert 'B: ["Array option",{"rank":2}]' in branches["choice"]
    assert "hidden-object-key" not in branches["choice"]
    assert "hidden-array-key" not in branches["choice"]
    assert 'A: {"level":"Low"}' in branches["score"]
    assert 'B: ["High",{"threshold":10}]' in branches["score"]


def test_null_noul_and_score_descriptions_fall_back_to_labels(compiler, payload):
    payload["questions"] = {
        "noul": {"type": "noul", "criteria": None},
        "score": {"type": "score", "criteria": [None, None]},
    }

    branches = {
        branch.question_id: compiler.tokenizer.decode(branch.input_ids)
        for branch in compiler.prepare(SystemOneRequest.model_validate(payload)).branches
    }

    assert "A: true\nB: false" in branches["noul"]
    assert "A: 0\nB: 1" in branches["score"]


def test_missing_instructions_falls_back_to_default(compiler, payload):
    del payload["questions"]["yes"]["instructions"]
    branch = compiler.prepare(SystemOneRequest.model_validate(payload)).branches[0]
    prompt = compiler.tokenizer.decode(branch.input_ids)
    assert f"Question: {DEFAULT_INSTRUCTIONS}" in prompt


@pytest.mark.parametrize("content", [[{"type": "image_url", "image_url": "http://x"}], 12])
def test_non_text_chat_is_rejected(content):
    with pytest.raises(ValueError):
        state_messages([{"role": "user", "content": content}])


def test_all_answer_labels_are_distinct_single_tokens(compiler):
    assert len(compiler.labels) == 64
    assert len({token_id for _, token_id in compiler.labels}) == 64
    for text, token_id in compiler.labels:
        assert compiler.tokenizer.encode(text) == [token_id]
