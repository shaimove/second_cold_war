from app import agents as agent_mod
from app.llm import LLMClient
from app.schemas import EvidenceSummary, DOMAIN_AGENTS


def test_evidence_agent_labels_hypothetical_in_mock_mode():
    llm = LLMClient()
    summary, n, _ = agent_mod.run_evidence_agent(
        llm,
        seed="If China enters a major financial crisis next year",
        scenario_mode="base_case",
    )
    assert isinstance(summary, EvidenceSummary)
    assert summary.hypothetical_assumptions
    assert n == 0  # empty KB in tests


def test_each_domain_agent_returns_required_fields():
    llm = LLMClient()
    for name in DOMAIN_AGENTS:
        out = agent_mod.run_domain_agent(
            llm,
            agent_name=name,
            seed="seed",
            scenario_mode="base_case",
            evidence_blob="evidence",
            round_number=1,
            previous_summary=None,
            previous_self_position=None,
        )
        assert out.agent_name == name
        assert out.main_assessment
        # At least one timeline contribution must exist.
        assert out.timeline_contributions


def test_security_agent_prompt_contains_safety_constraint():
    text = agent_mod.AGENT_SYSTEM_PROMPTS["security_taiwan"]
    assert "operational tactics" in text.lower()


def test_classify_event_status_marks_future_as_hypothetical():
    assert agent_mod.classify_event_status("If Taiwan elects ...") == "hypothetical"
    assert (
        agent_mod.classify_event_status("Suppose the U.S. announces export controls")
        == "hypothetical"
    )


def test_red_team_agent_returns_findings():
    llm = LLMClient()
    out, findings = agent_mod.run_red_team_agent(
        llm,
        seed="x",
        scenario_mode="base_case",
        evidence_blob="ev",
        final_round_summary=None,
    )
    assert out.agent_name == "red_team"
    assert isinstance(findings, list)
    assert len(findings) >= 1


def test_build_final_timeline_dedupes_and_covers_years():
    llm = LLMClient()
    outputs = {}
    for name in DOMAIN_AGENTS:
        outputs[name] = agent_mod.run_domain_agent(
            llm,
            agent_name=name,
            seed="seed",
            scenario_mode="base_case",
            evidence_blob="ev",
            round_number=1,
            previous_summary=None,
            previous_self_position=None,
        )
    timeline = agent_mod.build_final_timeline(outputs)
    years = [yb.year for yb in timeline]
    assert years == [2026, 2027, 2028, 2029, 2030, 2031]
