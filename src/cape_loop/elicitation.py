"""Matched provenance construction for identical-response audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .domains import DomainSpec
from .response import RandomUtilityModel
from .schemas import InteractionContext, LatentUser, Observation, Option


MECHANISMS = ("balanced", "restricted", "default", "suggested")


@dataclass(frozen=True, slots=True)
class MatchedAnchorSet:
    """Four contexts that preserve one anchor and isolate provenance mechanisms."""

    domain_id: str
    scenario_id: str
    target_attribute: int
    anchor_direction: int
    anchor_option_id: str
    contexts: Mapping[str, InteractionContext]

    def __post_init__(self) -> None:
        if set(self.contexts) != set(MECHANISMS):
            raise ValueError(f"contexts must contain exactly {MECHANISMS}")
        if self.anchor_direction not in (-1, 1):
            raise ValueError("anchor_direction must be -1 or +1")
        self.validate_invariants()

    def context(self, mechanism: str) -> InteractionContext:
        try:
            return self.contexts[mechanism]
        except KeyError as exc:
            raise KeyError(f"unknown matched mechanism: {mechanism}") from exc

    def anchor(self) -> Option:
        return self.context("balanced").option(self.anchor_option_id)

    def observation(self, surface_response: str | None = None) -> Observation:
        return Observation(
            selected_option_id=self.anchor_option_id,
            surface_response=surface_response,
            choice_noise_key=f"{self.scenario_id}:controlled-anchor",
        )

    def validate_invariants(self) -> None:
        contexts = self.contexts
        for mechanism, context in contexts.items():
            if context.domain != self.domain_id:
                raise ValueError(f"{mechanism} context has the wrong domain")
            if context.scenario_id != self.scenario_id:
                raise ValueError(f"{mechanism} context has the wrong scenario")
            if context.target_attribute != self.target_attribute:
                raise ValueError(f"{mechanism} context has the wrong target")
            if self.anchor_option_id not in context.option_ids:
                raise ValueError(f"{mechanism} omits the anchor")

        anchor = contexts["balanced"].option(self.anchor_option_id)
        for mechanism, context in contexts.items():
            if context.option(self.anchor_option_id) != anchor:
                raise ValueError(f"{mechanism} changed the anchor identity or attributes")

        balanced = contexts["balanced"]
        default = contexts["default"]
        suggested = contexts["suggested"]
        invariant_fields = (
            "options",
            "ranking",
            "domain",
            "scenario_id",
            "turn_id",
            "wording_template",
            "question_type",
            "target_attribute",
        )
        for mechanism, context in (("default", default), ("suggested", suggested)):
            for field_name in invariant_fields:
                if getattr(context, field_name) != getattr(balanced, field_name):
                    raise ValueError(
                        f"{mechanism} changes non-treatment field {field_name}"
                    )
        if balanced.default_option_id is not None or balanced.suggested_option_id is not None:
            raise ValueError("balanced context cannot contain a default or suggestion")
        if default.default_option_id != self.anchor_option_id:
            raise ValueError("default context must default the anchor")
        if default.suggested_option_id is not None:
            raise ValueError("default context also contains a suggestion")
        if suggested.suggested_option_id != self.anchor_option_id:
            raise ValueError("suggested context must suggest the anchor")
        if suggested.default_option_id is not None:
            raise ValueError("suggested context also contains a default")

        restricted = contexts["restricted"]
        restricted_other = next(
            option
            for option in restricted.options
            if option.option_id != self.anchor_option_id
        )
        anchor_feature = anchor.features[self.target_attribute]
        other_feature = restricted_other.features[self.target_attribute]
        if anchor_feature == 0 or other_feature == 0 or anchor_feature * other_feature <= 0:
            raise ValueError("restricted alternatives must share the target direction")
        balanced_other = next(
            option
            for option in balanced.options
            if option.option_id != self.anchor_option_id
        )
        if anchor_feature * balanced_other.features[self.target_attribute] >= 0:
            raise ValueError("balanced alternatives must oppose the anchor")

    def choice_probabilities(
        self,
        user: LatentUser,
        response_model: RandomUtilityModel,
    ) -> dict[str, float]:
        return {
            mechanism: response_model.probability_map(
                user.theta, user.susceptibility, context
            )[self.anchor_option_id]
            for mechanism, context in self.contexts.items()
        }

    def eligible(
        self,
        user: LatentUser,
        response_model: RandomUtilityModel,
        *,
        minimum_probability: float = 0.05,
        maximum_probability: float | None = None,
    ) -> bool:
        if not 0 < minimum_probability < 1:
            raise ValueError("minimum_probability must lie in (0, 1)")
        if (
            maximum_probability is not None
            and not minimum_probability < maximum_probability <= 1
        ):
            raise ValueError("maximum_probability must exceed the minimum")
        return all(
            probability > minimum_probability
            and (
                maximum_probability is None
                or probability < maximum_probability
            )
            for probability in self.choice_probabilities(user, response_model).values()
        )


def build_matched_anchor_set(
    domain: DomainSpec,
    *,
    target_attribute: int = 0,
    anchor_direction: int = -1,
    scenario_id: str = "anchor",
    wording_template: str = "neutral_matched_choice",
    turn: int = 0,
) -> MatchedAnchorSet:
    """Construct the paper's balanced/restricted/default/suggested quartet."""

    if anchor_direction not in (-1, 1):
        raise ValueError("anchor_direction must be -1 or +1")
    anchor = domain.directional_option(target_attribute, anchor_direction)
    opposite = domain.directional_option(target_attribute, -anchor_direction)
    peer_features = list(anchor.features)
    nuisance_attribute = (target_attribute + 1) % len(peer_features)
    peer_features[nuisance_attribute] = 0.25
    restricted_peer = Option(
        option_id=f"{anchor.option_id}_restricted_peer",
        features=(peer_features[0], peer_features[1], peer_features[2]),
        label=f"{anchor.label} alternative",
        domain=domain.domain_id,
    )
    common = {
        "domain": domain.domain_id,
        "scenario_id": scenario_id,
        "turn_id": str(turn),
        "wording_template": wording_template,
        "question_type": "choice",
        "target_attribute": target_attribute,
    }
    balanced_options = (anchor, opposite)
    ranking = (anchor.option_id, opposite.option_id)
    contexts = {
        "balanced": InteractionContext(
            context_id=f"{scenario_id}:balanced:{turn}",
            options=balanced_options,
            ranking=ranking,
            **common,
        ),
        "restricted": InteractionContext(
            context_id=f"{scenario_id}:restricted:{turn}",
            options=(anchor, restricted_peer),
            ranking=(anchor.option_id, restricted_peer.option_id),
            **common,
        ),
        "default": InteractionContext(
            context_id=f"{scenario_id}:default:{turn}",
            options=balanced_options,
            ranking=ranking,
            default_option_id=anchor.option_id,
            **common,
        ),
        "suggested": InteractionContext(
            context_id=f"{scenario_id}:suggested:{turn}",
            options=balanced_options,
            ranking=ranking,
            suggested_option_id=anchor.option_id,
            **common,
        ),
    }
    return MatchedAnchorSet(
        domain_id=domain.domain_id,
        scenario_id=scenario_id,
        target_attribute=target_attribute,
        anchor_direction=anchor_direction,
        anchor_option_id=anchor.option_id,
        contexts=contexts,
    )
