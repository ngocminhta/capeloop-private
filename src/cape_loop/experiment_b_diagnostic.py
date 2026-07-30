"""Bounded, live-capable diagnostic for the Experiment B closed loop.

This module deliberately fixes a small matched Experiment B comparison: one
user, one domain, one incorrect initial profile, two policies, and one
full-context LLM updater.  The turn count is limited to complete three-attribute
cycles, and the resulting two trajectories require exactly twice that many
logical completion-provider calls.  It is an execution diagnostic, not an
inferential experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .conversation_reporting import build_closed_loop_records
from .conversation_surfaces import ConversationTemplateBank
from .domains import DomainSpec
from .experiments.closed_loop import (
    ExperimentBResult,
    run_experiment_b,
)
from .llm_exchange import (
    CompletionProvider,
    LLMRequest,
    LLMResponse,
)
from .policies import BalancedPolicy, SoftProfileConditionedPolicy
from .response import RandomUtilityModel
from .scenarios import ScenarioCatalog
from .schemas import LatentUser
from .updaters import (
    ExactActionAwareUpdater,
    LLMReplayUpdater,
    UpdateViewKind,
)


ALLOWED_DIAGNOSTIC_TURNS = (3, 6, 9, 12)
DIAGNOSTIC_TURNS = 3
DIAGNOSTIC_POLICY_IDS = (
    "balanced",
    "soft_profile_conditioned",
)
DIAGNOSTIC_INITIAL_PROFILE_CONDITIONS = ("incorrect",)
DIAGNOSTIC_UPDATER_ID = "llm_full_context"


def expected_provider_calls(turns: int) -> int:
    """Return the logical-call invariant for one supported diagnostic."""

    if (
        not isinstance(turns, int)
        or isinstance(turns, bool)
        or turns not in ALLOWED_DIAGNOSTIC_TURNS
    ):
        allowed = ", ".join(str(value) for value in ALLOWED_DIAGNOSTIC_TURNS)
        raise ValueError(
            "turns must be a complete attribute cycle in "
            f"({allowed})"
        )
    return turns * len(DIAGNOSTIC_POLICY_IDS)


# Backward-compatible name for the default three-turn diagnostic.
EXPECTED_PROVIDER_CALLS = expected_provider_calls(DIAGNOSTIC_TURNS)


class _BoundedCallProvider:
    """Bound and audit the provider used by one diagnostic design."""

    def __init__(
        self,
        provider: CompletionProvider,
        *,
        expected_calls: int,
    ) -> None:
        self._provider = provider
        self._expected_calls = expected_calls
        self._requests: list[LLMRequest] = []
        self._responses: list[LLMResponse] = []

    @property
    def call_count(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> tuple[LLMRequest, ...]:
        return tuple(self._requests)

    @property
    def responses(self) -> tuple[LLMResponse, ...]:
        return tuple(self._responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.call_count >= self._expected_calls:
            raise RuntimeError(
                "Experiment B diagnostic permits exactly "
                f"{self._expected_calls} logical provider calls"
            )
        # Count attempted calls before delegating.  A provider exception must
        # not make an attempted live request disappear from local accounting.
        self._requests.append(request)
        response = self._provider.complete(request)
        if response.request_id != request.request_id:
            raise ValueError(
                "provider response request_id does not match request"
            )
        if response.prompt_sha256 != request.prompt_sha256:
            raise ValueError(
                "provider response prompt_sha256 does not match request"
            )
        self._responses.append(response)
        return response


@dataclass(frozen=True, slots=True)
class ExperimentBDiagnosticResult:
    """Native result plus renderer-ready records and provider exchange."""

    turns: int
    experiment_result: ExperimentBResult
    conversation_records: tuple[dict[str, Any], ...]
    requests: tuple[LLMRequest, ...]
    responses: tuple[LLMResponse, ...]
    provider_call_count: int

    def __post_init__(self) -> None:
        expected_calls = expected_provider_calls(self.turns)
        if self.provider_call_count != expected_calls:
            raise ValueError(
                "Experiment B diagnostic result must contain exactly "
                f"{expected_calls} logical provider calls"
            )
        if len(self.requests) != self.provider_call_count:
            raise ValueError(
                "provider call count and captured requests differ"
            )
        if len(self.responses) != self.provider_call_count:
            raise ValueError(
                "provider call count and captured responses differ"
            )


def run_experiment_b_diagnostic(
    *,
    user: LatentUser,
    domain: DomainSpec,
    provider: CompletionProvider,
    scenario_catalog: ScenarioCatalog,
    conversation_bank: ConversationTemplateBank,
    seed: int = 1729,
    data_split: str = "test",
    turns: int = DIAGNOSTIC_TURNS,
    response_model: RandomUtilityModel | None = None,
) -> ExperimentBDiagnosticResult:
    """Run one bounded Experiment B execution diagnostic.

    The evaluated model sees the full natural-language interaction.  A local
    exact action-aware updater consumes every realized event as the same-history
    shadow and makes no provider calls.
    """

    expected_calls = expected_provider_calls(turns)
    if not isinstance(user, LatentUser):
        raise TypeError("user must be a LatentUser")
    if not isinstance(domain, DomainSpec):
        raise TypeError("domain must be a DomainSpec")
    if not isinstance(scenario_catalog, ScenarioCatalog):
        raise TypeError("scenario_catalog must be a ScenarioCatalog")
    if not isinstance(conversation_bank, ConversationTemplateBank):
        raise TypeError(
            "conversation_bank must be a ConversationTemplateBank"
        )
    if scenario_catalog.catalog_status != "frozen-development":
        raise ValueError(
            "Experiment B diagnostic requires a frozen scenario catalog"
        )
    conversation_bank.validate_catalog(scenario_catalog)

    declared_response = response_model or RandomUtilityModel()
    guarded_provider = _BoundedCallProvider(
        provider,
        expected_calls=expected_calls,
    )
    updater = LLMReplayUpdater(
        DIAGNOSTIC_UPDATER_ID,
        UpdateViewKind.FULL_CONTEXT,
        guarded_provider,
    )
    policies = {
        policy.policy_id: policy
        for policy in (
            BalancedPolicy(),
            SoftProfileConditionedPolicy(),
        )
    }
    experiment_result = run_experiment_b(
        users=(user,),
        domains=(domain,),
        updaters={updater.updater_id: updater},
        policies=policies,
        initial_profile_conditions=(
            DIAGNOSTIC_INITIAL_PROFILE_CONDITIONS
        ),
        turns=turns,
        trajectories_per_cell=1,
        response_model=declared_response,
        shadow_updater=ExactActionAwareUpdater(declared_response),
        seed=seed,
        scenario_catalog=scenario_catalog,
        conversation_bank=conversation_bank,
        data_split=data_split,
    )

    if guarded_provider.call_count != expected_calls:
        raise AssertionError(
            "Experiment B diagnostic made "
            f"{guarded_provider.call_count} provider calls; expected "
            f"{expected_calls}"
        )
    if len(experiment_result.trajectories) != len(
        DIAGNOSTIC_POLICY_IDS
    ):
        raise AssertionError(
            "Experiment B diagnostic did not produce one trajectory per policy"
        )
    for trajectory in experiment_result.trajectories:
        if len(trajectory.turns) != turns:
            raise AssertionError(
                "Experiment B diagnostic trajectory has an unexpected "
                "turn count"
            )
        if not trajectory.same_history_shadow:
            raise AssertionError(
                "Experiment B diagnostic lost same-history shadow alignment"
            )

    model_ids = tuple(
        sorted({response.model_id for response in guarded_provider.responses})
    )
    conversation_records = build_closed_loop_records(
        experiment_result.trajectories,
        experiment="B",
        conversation_bank=conversation_bank,
        updater_views={
            DIAGNOSTIC_UPDATER_ID: UpdateViewKind.FULL_CONTEXT.value,
        },
        model_ids={DIAGNOSTIC_UPDATER_ID: model_ids},
        assessments=experiment_result.self_confirmation_assessments,
        comparisons=experiment_result.decompositions,
        split_by_user={user.user_id: data_split},
        extra_conditions={
            "diagnostic_only": True,
            "claim_eligible": False,
        },
    )
    if len(conversation_records) != len(experiment_result.trajectories):
        raise AssertionError(
            "Experiment B diagnostic conversation records are incomplete"
        )

    return ExperimentBDiagnosticResult(
        turns=turns,
        experiment_result=experiment_result,
        conversation_records=conversation_records,
        requests=guarded_provider.requests,
        responses=guarded_provider.responses,
        provider_call_count=guarded_provider.call_count,
    )


__all__ = [
    "ALLOWED_DIAGNOSTIC_TURNS",
    "DIAGNOSTIC_INITIAL_PROFILE_CONDITIONS",
    "DIAGNOSTIC_POLICY_IDS",
    "DIAGNOSTIC_TURNS",
    "DIAGNOSTIC_UPDATER_ID",
    "EXPECTED_PROVIDER_CALLS",
    "ExperimentBDiagnosticResult",
    "expected_provider_calls",
    "run_experiment_b_diagnostic",
]
