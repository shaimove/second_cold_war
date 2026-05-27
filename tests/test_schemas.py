import pytest

from app.schemas import (
    FinalScenario,
    ScenarioRequest,
    ScenarioState,
    TimelineEvent,
    YearBlock,
    empty_final_scenario,
)


def test_scenario_state_valid():
    s = ScenarioState(run_id="r1", seed="seed", scenario_mode="base_case")
    assert s.scenario_mode == "base_case"
    assert s.simulation_years == [2026, 2027, 2028, 2029, 2030, 2031]


def test_scenario_state_rejects_bad_mode():
    with pytest.raises(Exception):
        ScenarioState(run_id="r1", seed="seed", scenario_mode="not_a_mode")


def test_scenario_request_rejects_bad_mode():
    with pytest.raises(Exception):
        ScenarioRequest(seed="x", scenario_mode="bogus")


def test_timeline_event_probability_clamped():
    e = TimelineEvent(event="x", probability=99)
    assert 0 <= e.probability <= 1
    e2 = TimelineEvent(event="x", probability=-1)
    assert e2.probability == 0.0


def test_final_scenario_serializes():
    fs = FinalScenario(
        run_id="r1",
        seed="seed",
        scenario_mode="base_case",
        timeline=[YearBlock(year=2026)],
    )
    payload = fs.model_dump()
    assert payload["run_id"] == "r1"
    assert payload["timeline"][0]["year"] == 2026


def test_empty_final_scenario_shape():
    payload = empty_final_scenario("r1", "seed", "base_case")
    for key in (
        "run_id",
        "seed",
        "scenario_mode",
        "timeline",
        "agent_summaries",
        "image",
        "run_metrics",
    ):
        assert key in payload


def test_event_status_invalid_falls_back():
    s = ScenarioState(
        run_id="r1", seed="s", scenario_mode="base_case", event_status="weird"
    )
    assert s.event_status == "hypothetical"
