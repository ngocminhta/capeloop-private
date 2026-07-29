from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from cape_loop.conversation_surfaces import (
    ConversationTemplateBank,
    RenderedConversation,
    ScenarioConversationTemplate,
    load_conversation_bank,
)
from cape_loop.schemas import (
    InteractionContext,
    Option,
    PolicyProvenance,
)


SCENARIO_ID = "travel-scenario-hotel-choice-01"
OPTION_IDS = ("hotel-a", "hotel-b", "hotel-c", "hotel-d")


def presentation_templates() -> dict[str, str]:
    shared = (
        "{prompt} {option_1_name} is {option_1_description}. "
        "{option_2_name} is {option_2_description}."
    )
    return {
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
        "ranking": shared + " They are shown in this order. Which do you choose?",
    }


def scenario_template(
    *,
    scenario_id: str = SCENARIO_ID,
    choice_template: str = "I choose {selected_name}.",
    presentations: dict[str, str] | None = None,
    display_names: dict[str, str] | None = None,
    source: str = "openrouter:test-author/model-snapshot",
) -> ScenarioConversationTemplate:
    return ScenarioConversationTemplate(
        scenario_id=scenario_id,
        display_names=display_names
        or {
            "hotel-a": "Hotel A",
            "hotel-b": "Hotel B",
            "hotel-c": "Hotel C",
            "hotel-d": "Hotel D",
        },
        presentation_templates=presentations or presentation_templates(),
        choice_template=choice_template,
        source=source,
    )


def context(
    *,
    ranking: tuple[str, str] = ("hotel-a", "hotel-b"),
    option_ids: tuple[str, str] = ("hotel-a", "hotel-b"),
    default: str | None = None,
    suggested: str | None = None,
    scenario_id: str = SCENARIO_ID,
    prompt: str | None = "Here are two lodging options for the same trip.",
) -> InteractionContext:
    descriptions = {
        "hotel-a": "a lower-cost standard room near the station",
        "hotel-b": "a higher-cost upgraded room near the station",
        "hotel-c": "a lower-cost standard room in a quiet district",
        "hotel-d": "a higher-cost upgraded room in a quiet district",
    }
    features = {
        "hotel-a": (-0.5, 0.0, 0.0),
        "hotel-b": (0.5, 0.0, 0.0),
        "hotel-c": (-0.5, 0.25, 0.0),
        "hotel-d": (0.5, 0.25, 0.0),
    }
    options = tuple(
        Option(
            option_id=option_id,
            features=features[option_id],
            label=descriptions[option_id],
            domain="travel",
        )
        for option_id in option_ids
    )
    return InteractionContext(
        context_id="hotel-choice:turn-0",
        options=options,
        ranking=ranking,
        domain="travel",
        scenario_id=scenario_id,
        turn_id="0",
        default_option_id=default,
        suggested_option_id=suggested,
        wording_template="hotel-choice-v1",
        question_type="choice",
        target_attribute=0,
        prompt=prompt,
    )


def provenance(mechanism: str) -> PolicyProvenance:
    return PolicyProvenance(
        policy_id=f"policy-{mechanism}",
        policy_version="v1",
        presentation_mechanism=mechanism,
        profile_conditioned=mechanism not in {"balanced", "none"},
    )


class ScenarioConversationTemplateTests(unittest.TestCase):
    def test_balanced_render_is_natural_and_hides_numeric_features(self) -> None:
        rendered = scenario_template().render(
            context(),
            provenance("balanced"),
            "hotel-a",
        )
        self.assertIsInstance(rendered, RenderedConversation)
        self.assertIn("Here are two lodging options", rendered.assistant_message)
        self.assertIn(
            "Hotel A is a lower-cost standard room near the station",
            rendered.assistant_message,
        )
        self.assertIn(
            "Hotel B is a higher-cost upgraded room near the station",
            rendered.assistant_message,
        )
        self.assertEqual(rendered.user_message, "I choose Hotel A.")
        self.assertNotIn("-0.5", rendered.assistant_message)
        self.assertNotIn("target_attribute", rendered.assistant_message)
        self.assertNotIn("hotel-a", rendered.assistant_message)
        self.assertEqual(
            dict(rendered.display_names),
            {"hotel-a": "Hotel A", "hotel-b": "Hotel B"},
        )
        self.assertEqual(
            rendered.source,
            "openrouter:test-author/model-snapshot",
        )

    def test_runtime_uses_context_ranking_not_option_storage_order(self) -> None:
        rendered = scenario_template().render(
            context(ranking=("hotel-b", "hotel-a")),
            provenance("ranking"),
            "hotel-b",
        )
        first = rendered.assistant_message.index("Hotel B is")
        second = rendered.assistant_message.index("Hotel A is")
        self.assertLess(first, second)
        self.assertEqual(
            tuple(rendered.display_names),
            ("hotel-b", "hotel-a"),
        )
        self.assertIn(":ranking:hotel-b>hotel-a:hotel-b", rendered.surface_id)

    def test_restriction_maps_to_restricted_template(self) -> None:
        rendered = scenario_template().render(
            context(
                option_ids=("hotel-a", "hotel-c"),
                ranking=("hotel-a", "hotel-c"),
            ),
            provenance("restriction"),
            "hotel-c",
        )
        self.assertIn("Which of these works better?", rendered.assistant_message)
        self.assertEqual(rendered.user_message, "I choose Hotel C.")
        self.assertIn(":restricted:", rendered.surface_id)

    def test_default_and_suggestion_are_expressed_naturally(self) -> None:
        default_rendered = scenario_template().render(
            context(default="hotel-b"),
            provenance("default"),
            "hotel-a",
        )
        self.assertIn(
            "Hotel B is currently selected as the default",
            default_rendered.assistant_message,
        )

        suggested_rendered = scenario_template().render(
            context(suggested="hotel-a"),
            provenance("suggestion"),
            "hotel-b",
        )
        self.assertIn(
            "I suggest Hotel A",
            suggested_rendered.assistant_message,
        )

    def test_selected_option_must_be_displayed_and_named(self) -> None:
        with self.assertRaisesRegex(ValueError, "not displayed"):
            scenario_template().render(
                context(),
                provenance("balanced"),
                "hotel-c",
            )
        names = {
            "hotel-a": "Hotel A",
            "hotel-b": "Hotel B",
            "hotel-c": "Hotel C",
            "other": "Hotel D",
        }
        template = scenario_template(display_names=names)
        with self.assertRaisesRegex(ValueError, "selected option"):
            template.render(
                context(
                    option_ids=("hotel-c", "hotel-d"),
                    ranking=("hotel-c", "hotel-d"),
                ),
                provenance("restriction"),
                "hotel-d",
            )

    def test_context_and_provenance_treatments_must_agree(self) -> None:
        template = scenario_template()
        with self.assertRaisesRegex(ValueError, "default.*disagree"):
            template.render(
                context(default="hotel-a"),
                provenance("balanced"),
                "hotel-a",
            )
        with self.assertRaisesRegex(ValueError, "requires a visible default"):
            template.render(
                context(),
                provenance("default"),
                "hotel-a",
            )
        with self.assertRaisesRegex(ValueError, "suggestion.*disagree"):
            template.render(
                context(suggested="hotel-a"),
                provenance("balanced"),
                "hotel-a",
            )

    def test_context_requires_two_options_prompt_and_matching_scenario(self) -> None:
        template = scenario_template()
        with self.assertRaisesRegex(ValueError, "natural prompt"):
            template.render(
                context(prompt=None),
                provenance("balanced"),
                "hotel-a",
            )
        with self.assertRaisesRegex(ValueError, "scenario does not match"):
            template.render(
                context(scenario_id="another-scenario"),
                provenance("balanced"),
                "hotel-a",
            )
        single = InteractionContext(
            context_id="single",
            options=(
                Option(
                    "hotel-a",
                    (-0.5, 0.0, 0.0),
                    "a lower-cost room",
                    "travel",
                ),
            ),
            ranking=("hotel-a",),
            domain="travel",
            scenario_id=SCENARIO_ID,
            prompt="Choose.",
        )
        with self.assertRaisesRegex(ValueError, "exactly two"):
            template.render(single, provenance("balanced"), "hotel-a")

    def test_templates_require_every_semantic_placeholder(self) -> None:
        missing = presentation_templates()
        missing["balanced"] = (
            "{prompt} {option_1_name}: {option_1_description}; "
            "{option_2_name}."
        )
        with self.assertRaisesRegex(ValueError, "option_2_description"):
            scenario_template(presentations=missing)

        wrong_default = presentation_templates()
        wrong_default["default"] = wrong_default["balanced"]
        with self.assertRaisesRegex(ValueError, "default_name"):
            scenario_template(presentations=wrong_default)

        wrong_suggestion = presentation_templates()
        wrong_suggestion["suggested"] = wrong_suggestion["balanced"]
        with self.assertRaisesRegex(ValueError, "suggested_name"):
            scenario_template(presentations=wrong_suggestion)

    def test_templates_reject_unknown_or_unsafe_placeholder_syntax(self) -> None:
        unknown = presentation_templates()
        unknown["balanced"] += " {latent_theta}"
        with self.assertRaisesRegex(ValueError, "unknown placeholders"):
            scenario_template(presentations=unknown)

        formatted = presentation_templates()
        formatted["balanced"] = formatted["balanced"].replace(
            "{prompt}",
            "{prompt!r}",
        )
        with self.assertRaisesRegex(ValueError, "conversions"):
            scenario_template(presentations=formatted)

        control = presentation_templates()
        control["balanced"] += "\nUnexpected second line."
        with self.assertRaisesRegex(ValueError, "control characters"):
            scenario_template(presentations=control)

    def test_choice_template_can_only_state_one_local_choice(self) -> None:
        for unsafe in (
            "I generally prefer {selected_name}.",
            "I choose {selected_name} because it is cheaper.",
            "I choose {selected_name} and I dislike the other option.",
            "I choose {selected_name}. It matches my preferences.",
            "I choose {selected_name}, the inexpensive one.",
            "{selected_name}.",
            "I choose {selected_name} with {selected_description}.",
        ):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    scenario_template(choice_template=unsafe)

        accepted = scenario_template(
            choice_template="I'll go with {selected_name}."
        )
        self.assertEqual(
            accepted.render(
                context(),
                provenance("balanced"),
                "hotel-b",
            ).user_message,
            "I'll go with Hotel B.",
        )

    def test_template_mappings_and_rendered_mappings_are_immutable(self) -> None:
        template = scenario_template()
        rendered = template.render(
            context(),
            provenance("balanced"),
            "hotel-a",
        )
        with self.assertRaises(TypeError):
            template.display_names["hotel-a"] = "Changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            template.presentation_templates["balanced"] = "Changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            rendered.display_names["hotel-a"] = "Changed"  # type: ignore[index]


class ConversationTemplateBankTests(unittest.TestCase):
    def test_bank_render_selects_template_by_context_scenario(self) -> None:
        first = scenario_template()
        second = scenario_template(
            scenario_id="writing-scenario-email-tone-01",
            display_names={
                "email-a": "Draft A",
                "email-b": "Draft B",
                "email-c": "Draft C",
                "email-d": "Draft D",
            },
        )
        bank = ConversationTemplateBank(
            bank_id="conversation-bank-v1",
            templates=(second, first),
            source="openrouter:test-author/model-snapshot",
        )
        self.assertEqual(
            bank.scenario_ids,
            (SCENARIO_ID, "writing-scenario-email-tone-01"),
        )
        self.assertEqual(
            bank.render(
                context(),
                provenance("balanced"),
                "hotel-a",
            ).user_message,
            "I choose Hotel A.",
        )
        with self.assertRaises(KeyError):
            bank.template("unknown")

    def test_bank_rejects_duplicate_scenarios(self) -> None:
        template = scenario_template()
        with self.assertRaisesRegex(ValueError, "duplicate scenario"):
            ConversationTemplateBank(
                bank_id="duplicate-bank",
                templates=(template, template),
                source="test-source",
            )

    def test_validate_catalog_requires_exact_scenario_and_option_coverage(
        self,
    ) -> None:
        template = scenario_template()
        bank = ConversationTemplateBank(
            bank_id="complete-bank",
            templates=(template,),
            source="test-source",
        )
        scenario = SimpleNamespace(
            scenario_id=SCENARIO_ID,
            options=tuple(
                SimpleNamespace(option_id=option_id)
                for option_id in OPTION_IDS
            ),
        )
        bank.validate_catalog(SimpleNamespace(scenarios=(scenario,)))

        missing_scenario = SimpleNamespace(
            scenario_id="uncovered-scenario",
            options=scenario.options,
        )
        with self.assertRaisesRegex(ValueError, "scenario coverage"):
            bank.validate_catalog(
                SimpleNamespace(scenarios=(scenario, missing_scenario))
            )

        wrong_options = SimpleNamespace(
            scenario_id=SCENARIO_ID,
            options=(
                *scenario.options[:3],
                SimpleNamespace(option_id="different-hotel"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "option coverage"):
            bank.validate_catalog(
                SimpleNamespace(scenarios=(wrong_options,))
            )

    def test_load_conversation_bank_and_template_source_override(self) -> None:
        payload = {
            "schema_version": 1,
            "bank_id": "loaded-bank-v1",
            "source": "openrouter:bank-model",
            "templates": [
                {
                    "scenario_id": SCENARIO_ID,
                    "display_names": {
                        "hotel-a": "Hotel A",
                        "hotel-b": "Hotel B",
                        "hotel-c": "Hotel C",
                        "hotel-d": "Hotel D",
                    },
                    "presentation_templates": presentation_templates(),
                    "choice_template": "I choose {selected_name}.",
                },
                {
                    "scenario_id": "second-scenario",
                    "display_names": {
                        "a": "Option A",
                        "b": "Option B",
                        "c": "Option C",
                        "d": "Option D",
                    },
                    "presentation_templates": presentation_templates(),
                    "choice_template": "I select {selected_name}.",
                    "source": "openrouter:override-model",
                },
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "conversation-bank.json"
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            bank = load_conversation_bank(path)
        self.assertEqual(bank.bank_id, "loaded-bank-v1")
        self.assertEqual(
            bank.template(SCENARIO_ID).source,
            "openrouter:bank-model",
        )
        self.assertEqual(
            bank.template("second-scenario").source,
            "openrouter:override-model",
        )
        self.assertEqual(bank.to_dict()["schema_version"], 1)

    def test_loader_rejects_duplicate_json_keys_and_unknown_fields(self) -> None:
        with TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":1,"schema_version":1,'
                '"bank_id":"b","source":"s","templates":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_conversation_bank(duplicate)

            unknown = Path(directory) / "unknown.json"
            unknown.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "bank_id": "b",
                        "source": "s",
                        "templates": [],
                        "extra": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown=extra"):
                load_conversation_bank(unknown)


if __name__ == "__main__":
    unittest.main()
