from app.llm import LLMClient


def test_mock_text_runs_without_api_key():
    llm = LLMClient()
    assert llm.config.mock_mode is True
    text = llm.call_llm_text("sys", "user prompt", agent_name="geo_strategy")
    assert "[MOCK" in text
    assert llm.metrics.llm_calls == 1


def test_mock_json_returns_dict():
    llm = LLMClient()
    data = llm.call_llm_json("sys", "user", agent_name="geo_strategy", round_number=1)
    assert isinstance(data, dict)
    assert "main_assessment" in data
    assert llm.metrics.llm_calls == 1


def test_cache_hit_avoids_second_call():
    llm = LLMClient()
    llm.call_llm_json("sys", "same", agent_name="economy_technology", round_number=1)
    first_calls = llm.metrics.llm_calls
    llm.call_llm_json("sys", "same", agent_name="economy_technology", round_number=1)
    assert llm.metrics.cache_hits >= 1
    assert llm.metrics.llm_calls == first_calls  # no new live call


def test_cache_key_is_stable():
    llm = LLMClient()
    k1 = llm._cache_key("a", 1, "s", "u", {"x": 1})
    k2 = llm._cache_key("a", 1, "s", "u", {"x": 1})
    k3 = llm._cache_key("a", 2, "s", "u", {"x": 1})
    assert k1 == k2
    assert k1 != k3


def test_extract_json_fallback_on_invalid():
    from app.utils import extract_json

    assert extract_json("not json") is None
    assert extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json('garbage {"a": 2} trailing')["a"] == 2
