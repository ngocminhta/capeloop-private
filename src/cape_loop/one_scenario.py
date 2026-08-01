"""A compact, human-readable walkthrough of one hybrid-simulator event.

This module deliberately runs only one frozen scenario and one evaluated
full-context profile-writer call.  It is a debugging and explanation aid, not
an inferential experiment and not paper evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from .beliefs import PreferenceBelief
from .conversation_surfaces import ConversationTemplateBank
from .domains import get_domain
from .elicitation import MECHANISMS, build_matched_anchor_set
from .llm_exchange import CompletionProvider, LLMRequest, LLMResponse
from .metrics import (
    action_conditioned_update_error,
    marginal_brier,
    marginal_kl,
    marginal_l1,
)
from .response import RandomUtilityModel
from .scenarios import ScenarioCatalog, materialize_matched_anchor_set
from .schemas import LatentUser, Observation, PolicyProvenance, Susceptibility
from .updaters import (
    ExactActionAwareUpdater,
    LLMReplayUpdater,
    UpdateViewKind,
    make_update_view,
)


SCHEMA_VERSION = 1
DEMONSTRATION_STATUS: Mapping[str, Any] = {
    "status": "demonstration_only",
    "paper_eligible": False,
    "claim_eligible": False,
    "reason": (
        "A single synthetic scenario is useful for inspection and debugging "
        "but cannot support an empirical or inferential claim."
    ),
}
_PRESENTATION_MECHANISMS = {
    "balanced": "balanced",
    "restricted": "restriction",
    "ranking": "ranking",
    "default": "default",
    "suggested": "suggestion",
}


def _marginal_payload(belief: PreferenceBelief) -> dict[str, Any]:
    return belief.marginals().to_dict()


def _semantic_truth(user: LatentUser, domain_id: str) -> list[dict[str, Any]]:
    domain = get_domain(domain_id)
    result = []
    for attribute, value in zip(domain.attributes, user.theta):
        result.append(
            {
                "attribute": attribute.key,
                "direction": (
                    attribute.negative_label
                    if value < 0
                    else attribute.positive_label
                ),
                "strength": "strong" if abs(value) == 2 else "moderate",
            }
        )
    return result


class _ExactlyOneCallProvider:
    """Guard the public provider boundary and bind its response to the request."""

    def __init__(self, provider: CompletionProvider) -> None:
        self.provider = provider
        self.call_count = 0
        self.request: LLMRequest | None = None
        self.response: LLMResponse | None = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.call_count:
            raise RuntimeError(
                "the one-scenario workflow permits exactly one provider call"
            )
        self.call_count += 1
        response = self.provider.complete(request)
        if response.request_id != request.request_id:
            raise ValueError("provider response request_id does not match request")
        if response.prompt_sha256 != request.prompt_sha256:
            raise ValueError(
                "provider response prompt_sha256 does not match request"
            )
        self.request = request
        self.response = response
        return response


@dataclass(frozen=True, slots=True)
class OneScenarioResult:
    """Complete compact output of one explanatory hybrid-simulator run."""

    scenario_id: str
    domain_id: str
    split: str
    mechanism: str
    anchor_direction: int
    seed: int
    user_id: str
    semantic_user_profile: tuple[Mapping[str, Any], ...]
    assistant_message: str
    user_message: str
    selected_option_id: str
    selected_option_label: str
    choice_probability: float
    surface_id: str
    surface_source: str
    updater_id: str
    updater_view: str
    model_id: str
    provider_call_count: int
    model_system_instruction: str
    model_input: Mapping[str, Any]
    prior_marginals: Mapping[str, Any]
    exact_reference_marginals: Mapping[str, Any]
    evaluated_model_marginals: Mapping[str, Any]
    metrics: Mapping[str, float]

    @property
    def claim_status(self) -> Mapping[str, Any]:
        return dict(DEMONSTRATION_STATUS)

    def to_dict(self) -> dict[str, Any]:
        """Return the compact machine-readable JSON payload."""

        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "one_scenario_hybrid_walkthrough",
            "claim_status": dict(self.claim_status),
            "scenario": {
                "scenario_id": self.scenario_id,
                "domain": self.domain_id,
                "split": self.split,
                "mechanism": self.mechanism,
                "anchor_direction": self.anchor_direction,
            },
            "simulated_user": {
                "user_id": self.user_id,
                "choice_model": "deterministic_seeded_random_utility",
                "seed": self.seed,
                "semantic_profile": [
                    dict(item) for item in self.semantic_user_profile
                ],
            },
            "conversation": [
                {"role": "assistant", "content": self.assistant_message},
                {"role": "user", "content": self.user_message},
            ],
            "choice": {
                "selected_option_id": self.selected_option_id,
                "selected_option_label": self.selected_option_label,
                "selected_option_probability": self.choice_probability,
                "choice_source": "mathematical_user_simulator",
                "surface_id": self.surface_id,
                "surface_source": self.surface_source,
            },
            "evaluated_model": {
                "updater_id": self.updater_id,
                "updater_view": self.updater_view,
                "model_id": self.model_id,
                "provider_call_count": self.provider_call_count,
                "system_instruction": self.model_system_instruction,
                "model_input": self.model_input,
            },
            "profile_outputs": {
                "prior": self.prior_marginals,
                "exact_action_aware_reference": (
                    self.exact_reference_marginals
                ),
                "evaluated_full_context_model": (
                    self.evaluated_model_marginals
                ),
            },
            "metrics": dict(self.metrics),
        }

    def conversation_record(self) -> dict[str, Any]:
        """Return one canonical, compact ``conversation_trace`` record."""

        conversation_id = (
            f"demo:{self.scenario_id}:{self.mechanism}:{self.user_id}"
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "conversation_trace",
            "experiment": "demo",
            "conversation_id": conversation_id,
            "conversation_kind": "single_turn",
            "source_id": self.scenario_id,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "conditions": {
                "split": self.split,
                "mechanism": self.mechanism,
                "demonstration_only": True,
                "claim_eligible": False,
            },
            "dialogue": [
                {
                    "turn": 1,
                    "event_id": conversation_id,
                    "scenario_id": self.scenario_id,
                    "surface_id": self.surface_id,
                    "surface_available": True,
                    "assistant": self.assistant_message,
                    "user": self.user_message,
                    "selected_option_id": self.selected_option_id,
                    "selected_option_label": self.selected_option_label,
                    "presentation_mechanism": (
                        _PRESENTATION_MECHANISMS[self.mechanism]
                    ),
                    "choice_source": "mathematical_user_simulator",
                    "surface_source": self.surface_source,
                    "turn_metrics": {
                        "choice_probability": self.choice_probability,
                    },
                }
            ],
            "outcomes": [
                {
                    "updater_id": self.updater_id,
                    "updater_view": self.updater_view,
                    "model_ids": [self.model_id],
                    "metrics": dict(self.metrics),
                }
            ],
            "assessments": [],
            "comparisons": [],
        }

    def jsonl_records(self) -> tuple[dict[str, Any], ...]:
        """Return the single canonical record for ``conversation.jsonl``."""

        return (
            self.conversation_record(),
        )

    def render_markdown(self) -> str:
        return render_one_scenario_markdown(self)


def run_one_scenario(
    *,
    catalog: ScenarioCatalog,
    conversation_bank: ConversationTemplateBank,
    scenario_id: str,
    user: LatentUser,
    provider: CompletionProvider,
    mechanism: str = "balanced",
    anchor_direction: int = -1,
    seed: int = 1729,
    prior: PreferenceBelief | None = None,
    response_model: RandomUtilityModel | None = None,
    exact_susceptibilities: tuple[Susceptibility, ...] | None = None,
    updater_id: str = "llm_full_context",
) -> OneScenarioResult:
    """Run one natural conversation and exactly one evaluated LLM update.

    The mathematical response model chooses first.  The frozen conversation
    bank only verbalizes that choice.  The exact reference is local and makes
    no provider call; the evaluated full-context updater makes exactly one.
    """

    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    if anchor_direction not in (-1, 1):
        raise ValueError("anchor_direction must be -1 or +1")
    if not updater_id:
        raise ValueError("updater_id must be non-empty")
    scenario = catalog.scenario(scenario_id)
    if mechanism not in scenario.supported_mechanisms:
        raise ValueError(
            f"scenario {scenario_id!r} does not support {mechanism!r}"
        )
    # Validate the selected scenario without requiring callers to subset the
    # complete frozen bank or catalog.
    template = conversation_bank.template(scenario_id)
    if set(template.display_names) != {
        option.option_id for option in scenario.options
    }:
        raise ValueError(
            "conversation template option coverage differs from the scenario"
        )

    domain = get_domain(scenario.domain)
    matched = materialize_matched_anchor_set(
        build_matched_anchor_set(
            domain,
            target_attribute=scenario.target_attribute,
            anchor_direction=anchor_direction,
            scenario_id=scenario.scenario_id,
            wording_template=scenario.wording_template_id,
        ),
        scenario,
    )
    context = matched.context(mechanism)
    provenance = PolicyProvenance(
        policy_id=f"one_scenario_{mechanism}",
        policy_version="v1",
        random_seed=seed,
        presentation_mechanism=_PRESENTATION_MECHANISMS[mechanism],
        profile_conditioned=False,
    )
    declared_response = response_model or RandomUtilityModel()
    sampled = declared_response.sample(
        user.theta,
        user.susceptibility,
        context,
        seed,
        noise_key=(
            "one-scenario",
            scenario.scenario_id,
            mechanism,
            anchor_direction,
        ),
    )
    rendered = conversation_bank.render(
        context,
        provenance,
        sampled.selected_option_id,
    )
    observation = Observation(
        selected_option_id=sampled.selected_option_id,
        surface_response=rendered.user_message,
        choice_noise_key=sampled.choice_noise_key,
        assistant_message=rendered.assistant_message,
        surface_id=rendered.surface_id,
    )
    event_id = f"one-scenario:{scenario.scenario_id}:{mechanism}"
    initial = prior or PreferenceBelief.uniform()

    exact = ExactActionAwareUpdater(
        response_model=declared_response,
        susceptibilities=(
            exact_susceptibilities
            if exact_susceptibilities is not None
            else ExactActionAwareUpdater().susceptibilities
        ),
    )
    exact_result = exact.update(
        exact.initial_state(initial),
        make_update_view(
            exact.view_kind,
            context,
            observation,
            provenance,
            event_id=event_id,
        ),
    )
    guarded_provider = _ExactlyOneCallProvider(provider)
    evaluated = LLMReplayUpdater(
        updater_id,
        UpdateViewKind.FULL_CONTEXT,
        guarded_provider,
    )
    evaluated_result = evaluated.update(
        evaluated.initial_state(initial),
        make_update_view(
            evaluated.view_kind,
            context,
            observation,
            provenance,
            event_id=event_id,
        ),
    )
    if (
        guarded_provider.call_count != 1
        or guarded_provider.request is None
        or guarded_provider.response is None
    ):
        raise AssertionError(
            "one-scenario execution did not make exactly one provider call"
        )

    exact_after = exact_result.state.belief
    evaluated_after = evaluated_result.state.belief
    probabilities = declared_response.probability_map(
        user.theta,
        user.susceptibility,
        context,
    )
    metrics = {
        "action_conditioned_update_error": (
            action_conditioned_update_error(
                initial,
                evaluated_after,
                initial,
                exact_after,
            )
        ),
        "marginal_kl_from_exact_reference": marginal_kl(
            exact_after,
            evaluated_after,
        ),
        "evaluated_model_brier": marginal_brier(
            evaluated_after,
            user.theta,
        ),
        "exact_reference_brier": marginal_brier(
            exact_after,
            user.theta,
        ),
        "excess_brier_vs_exact_reference": (
            marginal_brier(evaluated_after, user.theta)
            - marginal_brier(exact_after, user.theta)
        ),
        "evaluated_update_magnitude": (
            marginal_l1(initial, evaluated_after) / 3.0
        ),
    }
    request = guarded_provider.request
    response = guarded_provider.response
    result = OneScenarioResult(
        scenario_id=scenario.scenario_id,
        domain_id=scenario.domain,
        split=scenario.split,
        mechanism=mechanism,
        anchor_direction=anchor_direction,
        seed=seed,
        user_id=user.user_id,
        semantic_user_profile=tuple(
            _semantic_truth(user, scenario.domain)
        ),
        assistant_message=rendered.assistant_message,
        user_message=rendered.user_message,
        selected_option_id=observation.selected_option_id,
        selected_option_label=context.option(
            observation.selected_option_id
        ).label,
        choice_probability=probabilities[observation.selected_option_id],
        surface_id=rendered.surface_id,
        surface_source=rendered.source,
        updater_id=updater_id,
        updater_view=UpdateViewKind.FULL_CONTEXT.value,
        model_id=response.model_id,
        provider_call_count=guarded_provider.call_count,
        model_system_instruction=request.system_instruction,
        model_input=dict(request.payload),
        prior_marginals=_marginal_payload(initial),
        exact_reference_marginals=_marginal_payload(exact_after),
        evaluated_model_marginals=_marginal_payload(evaluated_after),
        metrics=metrics,
    )
    # Fail here rather than writing a partially serializable result later.
    json.dumps(result.to_dict(), allow_nan=False)
    return result


def render_one_scenario_markdown(result: OneScenarioResult) -> str:
    """Render the same result as a concise, human-readable explanation."""

    metric_labels = {
        "action_conditioned_update_error": (
            "Update error vs exact action-aware reference ↓"
        ),
        "marginal_kl_from_exact_reference": (
            "Marginal KL from exact reference ↓"
        ),
        "evaluated_model_brier": "Evaluated-model profile Brier ↓",
        "exact_reference_brier": "Exact-reference profile Brier ↓",
        "excess_brier_vs_exact_reference": "Excess Brier vs reference ↓",
        "evaluated_update_magnitude": (
            "Evaluated-model update magnitude ↔"
        ),
    }
    lines = [
        "# One-scenario hybrid experiment walkthrough",
        "",
        "> **Demonstration only.** This single synthetic scenario is not "
        "paper-eligible and supports no empirical or inferential claim.",
        "",
        "## Natural conversation",
        "",
        "**Assistant**",
        "",
        *(
            f"> {line}" if line else ">"
            for line in result.assistant_message.splitlines()
        ),
        "",
        "**Simulated user**",
        "",
        *(
            f"> {line}" if line else ">"
            for line in result.user_message.splitlines()
        ),
        "",
        "The mathematical user simulator selected "
        f"**{result.selected_option_label}** "
        f"(`{result.selected_option_id}`); the frozen language bank then "
        "expressed that selection as the user sentence above.",
        "",
        "## Evaluated model",
        "",
        f"- Model: `{result.model_id}`",
        f"- Updater: `{result.updater_id}`",
        f"- View: `{result.updater_view}`",
        f"- Provider calls: **{result.provider_call_count}**",
        "- Model-visible interaction: the natural assistant and user messages, "
        "readable option descriptions, and the semantic preference schema",
        "",
        "The exact action-aware reference was computed locally and made no "
        "model-provider call.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {metric_labels[key]} | {value:.6f} |"
        for key, value in result.metrics.items()
    )
    lines.extend(
        [
            "",
            "Arrows show interpretation: ↓ lower is better; ↔ is descriptive "
            "rather than inherently better or worse.",
            "",
            "## Status",
            "",
            "- Scenario source: frozen synthetic catalog",
            f"- Conversation source: `{result.surface_source}`",
            "- Choice source: deterministic seeded mathematical simulator",
            "- Evidence status: demonstration/debugging only",
            "- Paper eligible: **no**",
            "- Claim eligible: **no**",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DEMONSTRATION_STATUS",
    "OneScenarioResult",
    "render_one_scenario_markdown",
    "run_one_scenario",
]
