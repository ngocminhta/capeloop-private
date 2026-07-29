from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import unittest

from cape_loop.beliefs import PreferenceBelief
from cape_loop.conversation_surfaces import (
    ConversationTemplateBank,
    ScenarioConversationTemplate,
)
from cape_loop.domains import TRAVEL, DomainSpec
from cape_loop.experiments.closed_loop import run_trajectory
from cape_loop.policies import PolicyAction
from cape_loop.schema_export import SCHEMAS
from cape_loop.schemas import (
    InteractionContext,
    LatentUser,
    Option,
    PolicyProvenance,
)
from cape_loop.updaters import (
    LLMReplayUpdater,
    NoUpdateUpdater,
    UpdateViewKind,
    make_update_view,
)


SCENARIO_ID = "travel-scenario-hotel-choice-01"


def _context() -> InteractionContext:
    return InteractionContext(
        context_id="hotel-choice:turn-0",
        options=(
            Option(
                option_id="hotel-a",
                features=(-0.5, 0.0, 0.0),
                label="a budget room near the station with breakfast included",
                domain="travel",
            ),
            Option(
                option_id="hotel-b",
                features=(0.5, 0.0, 0.0),
                label="a premium room near the station with a larger workspace",
                domain="travel",
            ),
        ),
        ranking=("hotel-a", "hotel-b"),
        domain="travel",
        scenario_id=SCENARIO_ID,
        turn_id="0",
        wording_template="hotel-choice-v1",
        question_type="choice",
        target_attribute=0,
        prompt="Choose one of these two hotels for the same trip.",
    )


def _provenance() -> PolicyProvenance:
    return PolicyProvenance(
        policy_id="fixture-balanced",
        policy_version="v1",
        presentation_mechanism="balanced",
        profile_conditioned=False,
    )


@dataclass(frozen=True)
class _StaticPolicy:
    context: InteractionContext
    policy_id: str = "fixture-balanced"
    policy_version: str = "v1"

    def action(
        self,
        domain: DomainSpec,
        belief: PreferenceBelief,
        *,
        turn: int,
        master_seed: int,
        trajectory_id: str,
    ) -> PolicyAction:
        del domain, belief, turn, master_seed, trajectory_id
        return PolicyAction(self.context, _provenance())


def _conversation_bank() -> ConversationTemplateBank:
    shared = (
        "{prompt} {option_1_name} is {option_1_description}. "
        "{option_2_name} is {option_2_description}."
    )
    template = ScenarioConversationTemplate(
        scenario_id=SCENARIO_ID,
        display_names={
            "hotel-a": "Hotel A",
            "hotel-b": "Hotel B",
            "hotel-c": "Hotel C",
            "hotel-d": "Hotel D",
        },
        presentation_templates={
            "balanced": shared + " Which one would you choose?",
            "restricted": shared + " Which of these works better?",
            "default": (
                shared
                + " {default_name} is currently selected as the default. "
                "Which one would you choose?"
            ),
            "suggested": (
                shared
                + " I suggest {suggested_name}. Which one would you choose?"
            ),
            "ranking": (
                shared
                + " They are shown in this order. Which one would you choose?"
            ),
        },
        choice_template="I choose {selected_name}.",
        source="fixture:reviewed-language",
    )
    return ConversationTemplateBank(
        bank_id="fixture-hybrid-bank",
        templates=(template,),
        source="fixture:reviewed-language",
    )


def _run(*, conversation_bank: ConversationTemplateBank | None):
    return run_trajectory(
        user=LatentUser("user-1", (2, -1, 1)),
        domain=TRAVEL,
        policy=_StaticPolicy(_context()),
        updater=NoUpdateUpdater(),
        turns=2,
        seed=20260729,
        trajectory_id="hybrid-integration",
        crn_key="shared-choice-noise",
        conversation_bank=conversation_bank,
    )


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_nested_keys(item) for item in value.values()),
            set(),
        )
    if isinstance(value, (list, tuple)):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


class _ProviderMustNotRun:
    def complete(self, request: Any) -> Any:
        del request
        raise AssertionError("integration request construction called a provider")


class HybridConversationRuntimeTests(unittest.TestCase):
    def test_public_schemas_cover_the_bank_and_run_manifest_input(self) -> None:
        bank_schema = SCHEMAS["conversation-template-bank"]
        template_schema = bank_schema["properties"]["templates"]["items"]
        self.assertEqual(
            set(
                template_schema["properties"][
                    "presentation_templates"
                ]["required"]
            ),
            {"balanced", "restricted", "default", "suggested", "ranking"},
        )
        run_inputs = SCHEMAS["run-manifest"]["properties"]["inputs"]
        self.assertIn(
            "conversation_templates",
            run_inputs["properties"],
        )

    def test_language_attachment_does_not_change_mathematical_choice(self) -> None:
        mathematical = _run(conversation_bank=None)
        hybrid = _run(conversation_bank=_conversation_bank())

        self.assertEqual(
            tuple(turn.selected_option_id for turn in mathematical.turns),
            tuple(turn.selected_option_id for turn in hybrid.turns),
        )
        self.assertTrue(
            all(
                interaction.observation.surface_response is None
                for interaction in mathematical.audit_record.interactions
            )
        )

        for turn, interaction in zip(
            hybrid.turns,
            hybrid.audit_record.interactions,
        ):
            observation = interaction.observation
            self.assertEqual(
                observation.selected_option_id,
                turn.selected_option_id,
            )
            selected_name = (
                "Hotel A"
                if observation.selected_option_id == "hotel-a"
                else "Hotel B"
            )
            self.assertEqual(
                observation.surface_response,
                f"I choose {selected_name}.",
            )
            self.assertIn(
                "Choose one of these two hotels",
                observation.assistant_message or "",
            )
            self.assertIn(
                "Hotel A is a budget room",
                observation.assistant_message or "",
            )
            self.assertIn(
                "Hotel B is a premium room",
                observation.assistant_message or "",
            )
            self.assertTrue(observation.surface_id)

        serialized = hybrid.audit_record.to_dict()
        serialized_observation = serialized["interactions"][0]["observation"]
        self.assertEqual(
            serialized_observation["assistant_message"],
            hybrid.audit_record.interactions[
                0
            ].observation.assistant_message,
        )
        self.assertEqual(
            serialized_observation["surface_response"],
            hybrid.audit_record.interactions[
                0
            ].observation.surface_response,
        )

    def test_full_context_request_contains_natural_conversation_only(self) -> None:
        interaction = _run(
            conversation_bank=_conversation_bank()
        ).audit_record.interactions[0]
        updater = LLMReplayUpdater(
            "llm_full_context",
            UpdateViewKind.FULL_CONTEXT,
            _ProviderMustNotRun(),
        )
        view = make_update_view(
            UpdateViewKind.FULL_CONTEXT,
            interaction.context,
            interaction.observation,
            interaction.provenance,
            event_id="full-context-request",
        )
        request = updater.build_request(
            updater.initial_state(PreferenceBelief.uniform()),
            view,
        )

        self.assertEqual(
            request.payload["context"]["conversation"],
            [
                {
                    "role": "assistant",
                    "content": interaction.observation.assistant_message,
                },
                {
                    "role": "user",
                    "content": interaction.observation.surface_response,
                },
            ],
        )
        profile_schema = request.payload["observation"]["profile_schema"]
        self.assertEqual(profile_schema["attribute_1"]["name"], "price")
        self.assertIn(
            "budget",
            profile_schema["attribute_1"]["values"]["-2"],
        )
        self.assertIn(
            "premium",
            profile_schema["attribute_1"]["values"]["+2"],
        )
        self.assertEqual(
            request.payload["context"]["options"][0],
            {
                "option_id": "hotel-a",
                "description": (
                    "a budget room near the station with breakfast included"
                ),
            },
        )
        prompt_keys = _nested_keys(request.payload)
        self.assertNotIn("features", prompt_keys)
        self.assertNotIn("target_attribute", prompt_keys)

    def test_response_only_is_an_explicit_conversation_ablation(self) -> None:
        interaction = _run(
            conversation_bank=_conversation_bank()
        ).audit_record.interactions[0]
        updater = LLMReplayUpdater(
            "llm_response_only",
            UpdateViewKind.RESPONSE_ONLY,
            _ProviderMustNotRun(),
        )
        view = make_update_view(
            UpdateViewKind.RESPONSE_ONLY,
            interaction.context,
            interaction.observation,
            interaction.provenance,
            event_id="response-only-request",
        )
        request = updater.build_request(
            updater.initial_state(PreferenceBelief.uniform()),
            view,
        )

        self.assertIsNone(view.context)
        self.assertIsNone(view.provenance)
        self.assertNotIn("context", request.payload)
        self.assertNotIn("conversation", _nested_keys(request.payload))
        self.assertEqual(
            request.payload["observation"]["user_message"],
            interaction.observation.surface_response,
        )
        self.assertIn(
            "profile_schema",
            request.payload["observation"],
        )
        self.assertNotIn("features", _nested_keys(request.payload))
        self.assertNotIn("target_attribute", _nested_keys(request.payload))


if __name__ == "__main__":
    unittest.main()
