from __future__ import annotations

from typing import Any, Mapping
from unittest.mock import patch
import json
import unittest

import cape_loop.openrouter_conversation_provider as provider_module
from cape_loop.conversation_surfaces import ScenarioConversationTemplate
from cape_loop.openrouter_conversation_provider import (
    BASE_TEMPLATE_PLACEHOLDERS,
    CHOICE_PLACEHOLDERS,
    DEFAULT_TREATMENT_SENTENCE,
    FIXED_CHOICE_TEMPLATE,
    OPENROUTER_API_KEY_ENV,
    PRESENTATION_KINDS,
    PRESENTATION_PLACEHOLDERS,
    SUGGESTED_TREATMENT_SENTENCE,
    ConversationBudgetExceeded,
    ConversationHTTPError,
    ConversationLiveExecutionRequired,
    ConversationMissingAPIKey,
    ConversationModelMismatch,
    ConversationResponseError,
    OpenRouterConversationConfig,
    OpenRouterConversationProvider,
    conversation_template_json_schema,
    prepare_conversation_request,
)
from cape_loop.openrouter_provider import HTTPResult
from cape_loop.scenarios import ScenarioOption, ScenarioSpec


MODEL = "google/gemini-3.6-flash"
TEST_KEY = "test-openrouter-secret"


def scenario(identifier: str = "travel-scenario-lodging") -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=identifier,
        family_id=identifier.replace("scenario", "family"),
        revision=1,
        status="provisional",
        split="test",
        domain="travel",
        task_family="lodging",
        target_attribute=0,
        target_key="price",
        difficulty="standard_tradeoff",
        prompt="Choose one lodging package for a short personal trip.",
        wording_template_id="hidden-test-template",
        negative_option=ScenarioOption(
            f"{identifier}-negative",
            "Lower-cost standard room in a mixed-use neighborhood",
            (-0.5, 0.0, 0.0),
        ),
        positive_option=ScenarioOption(
            f"{identifier}-positive",
            "Higher-cost upgraded room in a mixed-use neighborhood",
            (0.5, 0.0, 0.0),
        ),
        negative_same_direction_option=ScenarioOption(
            f"{identifier}-negative-peer",
            "Lower-cost standard room in a quiet outer neighborhood",
            (-0.5, 0.25, 0.0),
        ),
        positive_same_direction_option=ScenarioOption(
            f"{identifier}-positive-peer",
            "Higher-cost upgraded room in a quiet outer neighborhood",
            (0.5, 0.25, 0.0),
        ),
        supported_mechanisms=(
            "balanced",
            "restricted",
            "default",
            "suggested",
            "ranking",
            "suggestion",
        ),
        quality_assertions={
            "neutral_wording": True,
            "symmetric_surface": True,
            "no_treatment_cues": True,
            "no_split_cues": True,
            "no_real_entities": True,
            "no_time_sensitive_facts": True,
            "no_objective_dominance": True,
            "all_surface_facts_modeled_or_matched": True,
            "feature_role_contract": True,
        },
        review={
            "automated_validation": "pending",
            "surface_human_review": "not_completed",
            "scientific_human_review": "not_completed",
            "paper_eligible": False,
            "note": "Test fixture.",
        },
    )


def valid_content(spec: ScenarioSpec) -> dict[str, Any]:
    option_ids = [option.option_id for option in spec.options]
    return {
        "display_names": {
            option_id: f"Hotel {'ABCD'[index]}"
            for index, option_id in enumerate(option_ids)
        },
        "base_template": (
            "{prompt} Here are {option_1_name}, described as "
            "{option_1_description}, and {option_2_name}, described as "
            "{option_2_description}. Which hotel do you choose?"
        ),
    }


def response(
    spec: ScenarioSpec,
    *,
    content: Mapping[str, Any] | None = None,
    model: str = MODEL,
    status: int = 200,
) -> HTTPResult:
    raw = {
        "id": "gen_conversation_001",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        valid_content(spec) if content is None else content
                    ),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 300,
            "completion_tokens": 220,
            "total_tokens": 520,
        },
        "openrouter_metadata": {
            "endpoints": {
                "available": [
                    {
                        "provider": "Google AI Studio",
                        "model": model,
                        "selected": True,
                    }
                ]
            }
        },
    }
    return HTTPResult(
        status=status,
        headers={},
        body=json.dumps(raw).encode("utf-8"),
    )


class NoCredentialReads(dict[str, str]):
    def get(self, key: str, default: Any = None) -> Any:
        raise AssertionError(f"credential read during keyless operation: {key}")


class FakeTransport:
    def __init__(self, result: HTTPResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HTTPResult:
        self.calls.append(
            {
                "url": url,
                "body": body,
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        return self.result


class OpenRouterConversationPreparationTests(unittest.TestCase):
    def test_config_requires_a_pinned_model_and_small_budgets(self) -> None:
        for invalid in (
            "openrouter/auto",
            "google/gemini-flash-latest",
            "google/gemini-3.6-flash:free",
            "~google/gemini-3.6-flash",
            "gemini",
        ):
            with self.subTest(model=invalid):
                with self.assertRaises(ValueError):
                    OpenRouterConversationConfig(model=invalid)
        with self.assertRaises(ValueError):
            OpenRouterConversationConfig(model=MODEL, max_requests=129)
        with self.assertRaises(ValueError):
            OpenRouterConversationConfig(
                model=MODEL,
                max_output_tokens=2049,
            )

    def test_prepare_is_keyless_strict_and_leakage_limited(self) -> None:
        spec = scenario()
        config = OpenRouterConversationConfig(model=MODEL)
        with patch.object(
            provider_module.os,
            "environ",
            NoCredentialReads(),
        ):
            prepared = prepare_conversation_request(spec, config)

        self.assertNotIn("Authorization", prepared.headers)
        self.assertEqual(
            prepared.endpoint,
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertEqual(
            prepared.model_input,
            {
                "prompt": spec.prompt,
                "domain": spec.domain,
                "task_family": spec.task_family,
                "options": [
                    {
                        "option_id": option.option_id,
                        "label": option.label,
                    }
                    for option in spec.options
                ],
            },
        )
        visible = json.dumps(dict(prepared.model_input), sort_keys=True)
        for forbidden in (
            "scenario_id",
            "family_id",
            "split",
            "features",
            "target_attribute",
            "target_key",
            "theta",
            "susceptibility",
            "profile",
            "probability",
        ):
            self.assertNotIn(forbidden, visible)

        body = prepared.body
        self.assertEqual(body["model"], MODEL)
        self.assertNotIn("models", body)
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertEqual(body["provider"]["data_collection"], "deny")
        instruction = body["messages"][0]["content"]
        self.assertIn("scenario-appropriate concrete noun", instruction)
        self.assertIn('"Hotel A"', instruction)
        self.assertIn('"Draft A"', instruction)
        self.assertIn('"Route A"', instruction)
        self.assertIn("meaningful, natural assistant utterance", instruction)
        self.assertIn("Do not author treatment wording", instruction)
        self.assertNotIn("{treatment_sentence}", instruction)
        response_format = body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])

    def test_schema_requests_only_names_and_one_neutral_base_template(
        self,
    ) -> None:
        spec = scenario()
        schema = conversation_template_json_schema(spec)
        self.assertFalse(schema["additionalProperties"])
        option_schema = schema["properties"]["display_names"]
        self.assertEqual(
            set(option_schema["required"]),
            {option.option_id for option in spec.options},
        )
        self.assertFalse(option_schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            {"display_names", "base_template"},
        )
        self.assertNotIn("presentation_templates", schema["properties"])
        self.assertNotIn("choice_template", schema["properties"])
        description = schema["properties"]["base_template"]["description"]
        for placeholder in BASE_TEMPLATE_PLACEHOLDERS:
            self.assertIn("{" + placeholder + "}", description)
        self.assertNotIn("{treatment_sentence}", description)
        self.assertEqual(
            BASE_TEMPLATE_PLACEHOLDERS,
            provider_module.PAIR_PLACEHOLDERS,
        )
        self.assertEqual(CHOICE_PLACEHOLDERS, ("selected_name",))

    def test_provider_prepare_keeps_readable_non_checksum_log(self) -> None:
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(model=MODEL)
        )
        provider.prepare(scenario())
        self.assertEqual(len(provider.request_logs), 1)
        log = provider.request_logs[0]
        self.assertEqual(
            log["event"],
            "conversation_template_request_prepared",
        )
        self.assertIn("model_input", log)
        self.assertEqual(
            log["authored_fields"],
            ["display_names", "base_template"],
        )
        self.assertEqual(
            log["base_template_placeholders"],
            list(BASE_TEMPLATE_PLACEHOLDERS),
        )
        self.assertNotIn("sha", json.dumps(log).casefold())
        self.assertNotIn(TEST_KEY, json.dumps(log))


class OpenRouterConversationCompletionTests(unittest.TestCase):
    def test_live_authorization_precedes_any_credential_read(self) -> None:
        spec = scenario()
        transport = FakeTransport(response(spec))
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(model=MODEL),
            transport=transport,
        )
        with patch.object(
            provider_module.os,
            "environ",
            NoCredentialReads(),
        ):
            with self.assertRaises(ConversationLiveExecutionRequired):
                provider.complete(spec)
        self.assertFalse(transport.calls)

    def test_missing_key_fails_without_transport(self) -> None:
        spec = scenario()
        transport = FakeTransport(response(spec))
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=transport,
        )
        with patch.dict(provider_module.os.environ, {}, clear=True):
            with self.assertRaises(ConversationMissingAPIKey):
                provider.complete(spec)
        self.assertFalse(transport.calls)
        self.assertEqual(provider.request_count, 0)

    def test_complete_returns_bank_record_and_calls_once_per_scenario(
        self,
    ) -> None:
        spec = scenario()
        transport = FakeTransport(response(spec))
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                upstream_provider="google-ai-studio",
                live_execution=True,
            ),
            transport=transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            first = provider.complete(spec)
            second = provider.complete(spec)

        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(provider.request_count, 1)
        call = transport.calls[0]
        self.assertEqual(
            call["headers"]["Authorization"],
            f"Bearer {TEST_KEY}",
        )
        sent = json.loads(call["body"].decode("utf-8"))
        self.assertEqual(
            sent["provider"]["only"],
            ["google-ai-studio"],
        )
        self.assertFalse(sent["provider"]["allow_fallbacks"])

        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["scenario_id"], spec.scenario_id)
        self.assertEqual(
            set(first["display_names"]),
            {option.option_id for option in spec.options},
        )
        self.assertEqual(
            set(first["presentation_templates"]),
            set(PRESENTATION_KINDS),
        )
        presentations = first["presentation_templates"]
        self.assertEqual(
            presentations["balanced"],
            valid_content(spec)["base_template"],
        )
        self.assertEqual(
            presentations["balanced"],
            presentations["restricted"],
        )
        self.assertEqual(
            presentations["balanced"],
            presentations["ranking"],
        )
        self.assertNotIn(
            "{treatment_sentence}",
            presentations["balanced"],
        )
        self.assertIn(
            DEFAULT_TREATMENT_SENTENCE,
            presentations["default"],
        )
        self.assertIn(
            SUGGESTED_TREATMENT_SENTENCE,
            presentations["suggested"],
        )
        self.assertEqual(
            presentations["default"],
            valid_content(spec)["base_template"].replace(
                "{prompt}",
                "{prompt} " + DEFAULT_TREATMENT_SENTENCE,
                1,
            ),
        )
        self.assertEqual(
            presentations["suggested"],
            valid_content(spec)["base_template"].replace(
                "{prompt}",
                "{prompt} " + SUGGESTED_TREATMENT_SENTENCE,
                1,
            ),
        )
        for kind in PRESENTATION_KINDS:
            for placeholder in PRESENTATION_PLACEHOLDERS[kind]:
                self.assertEqual(
                    presentations[kind].count(
                        "{" + placeholder + "}"
                    ),
                    1,
                )
        self.assertEqual(
            first["choice_template"],
            FIXED_CHOICE_TEMPLATE,
        )
        generator = first["generator"]
        self.assertEqual(generator["provider"], "openrouter")
        self.assertEqual(generator["model_requested"], MODEL)
        self.assertEqual(generator["model_returned"], MODEL)
        self.assertEqual(
            generator["upstream_provider"],
            "Google AI Studio",
        )
        self.assertFalse(generator["allow_fallbacks"])
        self.assertEqual(generator["validation_status"], "passed")
        self.assertEqual(
            generator["authored_fields"],
            ["display_names", "base_template"],
        )
        self.assertEqual(
            generator["treatment_expansion"],
            "local-fixed-v1",
        )

        # The locally expanded output is accepted by the core bank contract.
        core_template = ScenarioConversationTemplate(
            scenario_id=first["scenario_id"],
            display_names=first["display_names"],
            presentation_templates=first["presentation_templates"],
            choice_template=first["choice_template"],
            source=f"openrouter:{MODEL}:test",
        )
        self.assertEqual(
            core_template.presentation_templates["balanced"],
            core_template.presentation_templates["restricted"],
        )

        self.assertEqual(len(provider.result_logs), 1)
        self.assertEqual(
            provider.result_logs[0]["authored_base_template"],
            valid_content(spec)["base_template"],
        )
        self.assertEqual(
            provider.result_logs[0]["authored_display_names"],
            valid_content(spec)["display_names"],
        )
        serialized_logs = json.dumps(
            {
                "requests": provider.request_logs,
                "results": provider.result_logs,
            }
        )
        self.assertNotIn(TEST_KEY, serialized_logs)
        self.assertNotIn("sha", serialized_logs.casefold())

    def test_wrong_model_is_rejected(self) -> None:
        spec = scenario()
        transport = FakeTransport(
            response(spec, model="anthropic/claude-sonnet-5")
        )
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            with self.assertRaises(ConversationModelMismatch):
                provider.complete(spec)
        self.assertEqual(len(transport.calls), 1)

    def test_http_error_is_not_retried_and_redacts_secret(self) -> None:
        spec = scenario()
        body = json.dumps(
            {"error": {"message": f"bad credential {TEST_KEY}"}}
        ).encode("utf-8")
        transport = FakeTransport(
            HTTPResult(status=500, headers={}, body=body)
        )
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            with self.assertRaises(ConversationHTTPError) as raised:
                provider.complete(spec)
        self.assertEqual(len(transport.calls), 1)
        self.assertNotIn(TEST_KEY, str(raised.exception))
        self.assertIn("[redacted]", str(raised.exception))

    def test_budget_blocks_a_second_distinct_scenario_before_transport(
        self,
    ) -> None:
        first_spec = scenario("travel-scenario-one")
        second_spec = scenario("travel-scenario-two")
        transport = FakeTransport(response(first_spec))
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                max_requests=1,
                live_execution=True,
            ),
            transport=transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            provider.complete(first_spec)
            with self.assertRaises(ConversationBudgetExceeded):
                provider.complete(second_spec)
        self.assertEqual(len(transport.calls), 1)

    def test_invalid_base_placeholder_and_model_authored_treatment_rejected(
        self,
    ) -> None:
        spec = scenario()
        invalid = valid_content(spec)
        invalid["base_template"] = invalid["base_template"].replace(
            "{option_2_description}",
            "{invented_fact}",
        )
        transport = FakeTransport(response(spec, content=invalid))
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ConversationResponseError,
                "must use exactly",
            ):
                provider.complete(spec)

        invalid_treatment = valid_content(spec)
        invalid_treatment["base_template"] = (
            invalid_treatment["base_template"].replace(
                "Here are",
                "The default choices are",
            )
        )
        second_transport = FakeTransport(
            response(spec, content=invalid_treatment)
        )
        second_provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=second_transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ConversationResponseError,
                "treatment cue",
            ):
                second_provider.complete(spec)

    def test_model_cannot_author_choice_or_multiple_presentations(self) -> None:
        spec = scenario()
        invalid = valid_content(spec)
        invalid["choice_template"] = FIXED_CHOICE_TEMPLATE
        invalid["presentation_templates"] = {
            kind: invalid["base_template"]
            for kind in PRESENTATION_KINDS
        }
        transport = FakeTransport(response(spec, content=invalid))
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ConversationResponseError,
                "missing or unknown fields",
            ):
                provider.complete(spec)

    def test_legacy_treatment_placeholder_is_rejected(self) -> None:
        spec = scenario()
        invalid = valid_content(spec)
        invalid["base_template"] = invalid["base_template"].replace(
            "Which hotel",
            "{treatment_sentence} Which hotel",
        )
        transport = FakeTransport(response(spec, content=invalid))
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ConversationResponseError,
                "must use exactly",
            ):
                provider.complete(spec)

    def test_option_names_must_be_neutral_and_share_one_stem(self) -> None:
        spec = scenario()
        invalid = valid_content(spec)
        invalid["display_names"] = dict(invalid["display_names"])
        invalid["display_names"][spec.options[1].option_id] = "Room B"
        transport = FakeTransport(response(spec, content=invalid))
        provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ConversationResponseError,
                "same neutral noun stem",
            ):
                provider.complete(spec)

    def test_generic_names_and_statistical_base_are_rejected(self) -> None:
        spec = scenario()
        generic = valid_content(spec)
        generic["display_names"] = {
            option.option_id: f"Option {'ABCD'[index]}"
            for index, option in enumerate(spec.options)
        }
        generic_transport = FakeTransport(
            response(spec, content=generic)
        )
        generic_provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=generic_transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ConversationResponseError,
                "scenario-appropriate concrete noun",
            ):
                generic_provider.complete(spec)

        statistical = valid_content(spec)
        statistical["base_template"] = (
            statistical["base_template"].replace(
                "Here are",
                "Using the feature vector and probability, here are",
            )
        )
        statistical_transport = FakeTransport(
            response(spec, content=statistical)
        )
        statistical_provider = OpenRouterConversationProvider(
            OpenRouterConversationConfig(
                model=MODEL,
                live_execution=True,
            ),
            transport=statistical_transport,
        )
        with patch.dict(
            provider_module.os.environ,
            {OPENROUTER_API_KEY_ENV: TEST_KEY},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ConversationResponseError,
                "mathematical or statistical surface",
            ):
                statistical_provider.complete(spec)


if __name__ == "__main__":
    unittest.main()
