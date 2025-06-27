"""Tests for prompt modules."""


def test_answer_evaluation_prompts_import():
    """Test that answer evaluation prompts can be imported and contain expected content."""
    from karenina.prompts.answer_evaluation import ANSWER_EVALUATION_SYS, ANSWER_EVALUATION_USER

    # Test that the prompts are not empty
    assert len(ANSWER_EVALUATION_SYS) > 0
    assert len(ANSWER_EVALUATION_USER) > 0

    # Test that they contain expected keywords
    assert "role" in ANSWER_EVALUATION_SYS.lower()
    assert "json" in ANSWER_EVALUATION_SYS.lower()
    assert "question" in ANSWER_EVALUATION_USER
    assert "response" in ANSWER_EVALUATION_USER

    # Test that placeholders are present
    assert "{question}" in ANSWER_EVALUATION_USER
    assert "{response}" in ANSWER_EVALUATION_USER


def test_answer_generation_prompts_import():
    """Test that answer generation prompts can be imported and contain expected content."""
    from karenina.prompts.answer_generation import ANSWER_GENERATION_SYS, ANSWER_GENERATION_USER

    # Test that the prompts are not empty
    assert len(ANSWER_GENERATION_SYS) > 0
    assert len(ANSWER_GENERATION_USER) > 0

    # Test that they are strings
    assert isinstance(ANSWER_GENERATION_SYS, str)
    assert isinstance(ANSWER_GENERATION_USER, str)

    # Test that placeholders are present in user prompt
    assert "{question}" in ANSWER_GENERATION_USER
    assert "{question_json}" in ANSWER_GENERATION_USER


def test_prompt_formatting():
    """Test that prompts can be formatted with sample data."""
    from karenina.prompts.answer_evaluation import ANSWER_EVALUATION_USER
    from karenina.prompts.answer_generation import ANSWER_GENERATION_USER

    # Test answer evaluation formatting
    eval_formatted = ANSWER_EVALUATION_USER.format(question="Sample question?", response="Sample response")
    assert "Sample question?" in eval_formatted
    assert "Sample response" in eval_formatted

    # Test answer generation formatting
    gen_formatted = ANSWER_GENERATION_USER.format(
        question="Sample question?", question_json='{"id": "test", "question": "Sample question?"}'
    )
    assert "Sample question?" in gen_formatted
    assert '"id": "test"' in gen_formatted
