"""Pydantic schemas for state and final output.

These types define the contract between agents, the graph, and the API.
They are intentionally permissive (most fields default-empty) so the
mock pipeline and partial failures still produce a serializable result.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


SCENARIO_MODES = ("base_case", "escalation", "de_escalation", "wildcard")
EVENT_STATUSES = ("observed", "hypothetical", "mixed")
DOMAINS = (
    "economy",
    "strategy",
    "technology",
    "security",
    "ideology",
    "historical",
)
IMPACT_LEVELS = ("low", "medium", "high")
CONFIDENCE_LEVELS = ("low", "medium", "high")
AGENT_NAMES = (
    "orchestrator",
    "evidence_rag",
    "geo_strategy",
    "economy_technology",
    "domestic_ideology",
    "security_taiwan",
    "historical_analogy",
    "red_team",
)
DOMAIN_AGENTS = (
    "geo_strategy",
    "economy_technology",
    "domestic_ideology",
    "security_taiwan",
    "historical_analogy",
)


class EvidenceChunk(BaseModel):
    source_path: str = ""
    source_type: str = "unknown"
    domain: str = "general"
    text: str = ""
    score: float = 0.0


class EvidenceSummary(BaseModel):
    observed_facts: List[str] = Field(default_factory=list)
    historical_analogies: List[str] = Field(default_factory=list)
    strategy_frameworks: List[str] = Field(default_factory=list)
    hypothetical_assumptions: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    compact_summary: str = ""
    note: str = ""


class TimelineEvent(BaseModel):
    event: str = ""
    domain: str = "strategy"
    probability: float = 0.5
    impact: str = "medium"
    confidence: str = "medium"
    rationale: str = ""

    @field_validator("probability")
    @classmethod
    def _clamp_probability(cls, v: float) -> float:
        try:
            v = float(v)
        except Exception:
            return 0.5
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v


class YearBlock(BaseModel):
    year: int
    headline: str = ""
    events: List[TimelineEvent] = Field(default_factory=list)


class AgentTimelineContribution(BaseModel):
    year: int
    event: str = ""
    probability: float = 0.5
    impact: str = "medium"
    confidence: str = "medium"
    rationale: str = ""


class AgentOutput(BaseModel):
    agent_name: str
    round_number: int = 1
    main_assessment: str = ""
    key_drivers: List[str] = Field(default_factory=list)
    timeline_contributions: List[AgentTimelineContribution] = Field(
        default_factory=list
    )
    risks: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    agreements: List[str] = Field(default_factory=list)
    disagreements: List[str] = Field(default_factory=list)
    position_changed_from_previous_round: bool = False


class DiscussionSummary(BaseModel):
    round_number: int
    areas_of_agreement: List[str] = Field(default_factory=list)
    areas_of_disagreement: List[str] = Field(default_factory=list)
    emerging_timeline: List[str] = Field(default_factory=list)
    key_uncertainties: List[str] = Field(default_factory=list)
    agent_positions: Dict[str, str] = Field(default_factory=dict)


class RedTeamFinding(BaseModel):
    issue: str
    severity: str = "medium"
    affected_assumption: str = ""


class RunMetrics(BaseModel):
    llm_calls: int = 0
    agents_used: List[str] = Field(default_factory=list)
    retrieved_docs: int = 0
    cache_hits: int = 0
    discussion_rounds_completed: int = 0
    elapsed_seconds: float = 0.0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0


class ImageResult(BaseModel):
    enabled: bool = False
    generated: bool = False
    path: Optional[str] = None
    error: Optional[str] = None
    mock: bool = False


class ScenarioState(BaseModel):
    """Mutable state passed between LangGraph nodes."""

    run_id: str
    seed: str
    scenario_mode: str = "base_case"
    event_status: str = "hypothetical"
    current_year: int = 2026
    simulation_years: List[int] = Field(
        default_factory=lambda: [2026, 2027, 2028, 2029, 2030, 2031]
    )
    evidence_summary: EvidenceSummary = Field(default_factory=EvidenceSummary)
    discussion_rounds: List[DiscussionSummary] = Field(default_factory=list)
    agent_outputs: Dict[str, List[AgentOutput]] = Field(default_factory=dict)
    disagreements: List[str] = Field(default_factory=list)
    red_team_findings: List[RedTeamFinding] = Field(default_factory=list)
    final_timeline: List[YearBlock] = Field(default_factory=list)
    scenario_title: str = ""
    scenario_summary: str = ""
    image_prompt: str = ""
    image_result: ImageResult = Field(default_factory=ImageResult)
    run_metrics: RunMetrics = Field(default_factory=RunMetrics)
    errors: List[str] = Field(default_factory=list)

    @field_validator("scenario_mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in SCENARIO_MODES:
            raise ValueError(
                "scenario_mode must be one of: " + ", ".join(SCENARIO_MODES)
            )
        return v

    @field_validator("event_status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in EVENT_STATUSES:
            return "hypothetical"
        return v


class ScenarioRequest(BaseModel):
    seed: str
    scenario_mode: str = "base_case"

    @field_validator("scenario_mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in SCENARIO_MODES:
            raise ValueError(
                "scenario_mode must be one of: " + ", ".join(SCENARIO_MODES)
            )
        return v


class FinalScenario(BaseModel):
    """Public-facing output returned to the frontend and saved to DB."""

    run_id: str
    scenario_title: str = ""
    scenario_summary: str = ""
    seed: str
    scenario_mode: str
    event_status: str = "hypothetical"
    timeline: List[YearBlock] = Field(default_factory=list)
    key_assumptions: List[str] = Field(default_factory=list)
    main_disagreements: List[str] = Field(default_factory=list)
    red_team_warnings: List[str] = Field(default_factory=list)
    agent_summaries: Dict[str, str] = Field(default_factory=dict)
    discussion_summary: List[DiscussionSummary] = Field(default_factory=list)
    image_prompt: str = ""
    image: ImageResult = Field(default_factory=ImageResult)
    run_metrics: RunMetrics = Field(default_factory=RunMetrics)


class SavedRunSummary(BaseModel):
    run_id: str
    created_at: str
    seed: str
    scenario_mode: str
    scenario_title: str


def empty_final_scenario(
    run_id: str, seed: str, scenario_mode: str
) -> Dict[str, Any]:
    """A safe default shape used by error paths and fallbacks."""
    return FinalScenario(
        run_id=run_id, seed=seed, scenario_mode=scenario_mode
    ).model_dump()
