from __future__ import annotations

import unittest

from cape_loop.decoder_study import (
    DecoderTruthLabel,
    ExternalDecoderJudgment,
    ExternalDecoderRequest,
    HumanCollectionRecord,
)
from cape_loop.heldout import (
    HeldOutParaphraseCase,
    ParaphraseEvaluationRecord,
    ParaphraseTransferCriterion,
)
from cape_loop.llm_exchange import LLMResponse
from cape_loop.openai_provider import OpenAIProviderResult
from cape_loop.schema_export import SCHEMAS


_DIGEST = "a" * 64
_ROWS = (
    (0.25, 0.25, 0.25, 0.25),
    (0.25, 0.25, 0.25, 0.25),
    (0.25, 0.25, 0.25, 0.25),
)


class ExternalSchemaTests(unittest.TestCase):
    def assert_matches_top_level_contract(
        self, schema_name: str, record: dict[str, object]
    ) -> None:
        schema = SCHEMAS[schema_name]
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertTrue(set(schema["required"]).issubset(record))
        self.assertTrue(set(record).issubset(schema["properties"]))

    def test_external_records_match_exported_field_contracts(self) -> None:
        request = ExternalDecoderRequest.build(
            request_id="decoder-request-1",
            pseudonymous_state_id="state-pseudonym-1",
            representation_id="blinded-native-content-v1",
            evaluation_split="test",
            payload={"persona_text": "A blinded content-only representation."},
        )
        judgment = ExternalDecoderJudgment(
            request_id=request.request_id,
            request_sha256=request.request_sha256,
            decoder_instance_id="decoder-instance-1",
            decoder_family_id="decoder-family-1",
            judgment_origin="external_model",
            source_descriptor="externally supplied test fixture",
            blind_to_system_identity=True,
            blind_to_latent_truth=True,
            probabilities=_ROWS,
        )
        truth = DecoderTruthLabel(
            pseudonymous_state_id=request.pseudonymous_state_id,
            theta=(-2, -1, 1),
            evaluation_split="test",
        )
        human = HumanCollectionRecord(
            participant_code="participant-pseudonym-1",
            assignment_id="assignment-1",
            assignment_protocol_id="protocol-v1",
            display_id="display-1",
            rating=4,
            response_time_ms=1234,
            consent_version="consent-v1",
            consented=True,
            blinding_version="blinding-v1",
            comprehension_check_id="check-1",
            comprehension_passed=True,
        )
        case = HeldOutParaphraseCase(
            case_id="trial-1:surface:test-template-1",
            source_trial_id="trial-1",
            domain_id="travel",
            mechanism="balanced",
            selected_option_id="travel-option-1",
            template_id="test-template-1",
            family_id="test-family-1",
            split="test",
            template_sha256=_DIGEST,
            context_sha256=_DIGEST,
            surface_response="Given only these choices, this option works.",
        )
        evaluation = ParaphraseEvaluationRecord.from_case(
            case,
            updater_id="fitted_action_aware",
            brier=0.25,
            belief_payload={"attribute_1": [0.25, 0.25, 0.25, 0.25]},
        )
        criterion = ParaphraseTransferCriterion(
            verified=None,
            complete=False,
            material_gap=0.01,
            required_mechanisms=2,
            covered_domains=("travel",),
            covered_template_ids=(case.template_id,),
            expected_template_ids=(case.template_id, "test-template-2"),
            qualifying_mechanisms=(),
            mean_gaps=(),
            missing_pairs=("test-template-2:writing:restricted",),
        )

        records = {
            "external-decoder-request": request.to_dict(),
            "external-decoder-judgment": judgment.to_dict(),
            "decoder-truth-label": truth.to_dict(),
            "human-collection": human.to_dict(),
            "heldout-paraphrase-case": case.to_dict(),
            "heldout-paraphrase-evaluation": evaluation.to_dict(),
            "heldout-paraphrase-criterion": criterion.to_dict(),
        }
        for name, record in records.items():
            with self.subTest(schema=name):
                self.assert_matches_top_level_contract(name, record)

    def test_openai_audit_embeds_provider_neutral_replay_response(self) -> None:
        beliefs = {
            f"attribute_{attribute}": {
                "-2": 0.25,
                "-1": 0.25,
                "+1": 0.25,
                "+2": 0.25,
            }
            for attribute in (1, 2, 3)
        }
        response = LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": "profile-request-1",
                "prompt_sha256": _DIGEST,
                "model_id": "provider-model",
                "beliefs": beliefs,
                "raw_response_sha256": _DIGEST,
            }
        )
        result = OpenAIProviderResult(
            response=response,
            model_requested="requested-model",
            model_returned="provider-model",
            provider_response_id="response-1",
            provider_created_at=1,
            usage={"total_tokens": 42},
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            attempts=1,
            request_body_sha256=_DIGEST,
            idempotency_key="cape-loop-idempotency",
            client_request_id="cape-loop-client-request",
            server_request_id="server-request-1",
            processing_ms="10",
            estimated_max_tokens=100,
            raw_response={"id": "response-1"},
        )
        audit = result.to_audit_record()

        self.assert_matches_top_level_contract("openai-provider-audit", audit)
        self.assertEqual(audit["replay_response"], response.to_dict())
        embedded = SCHEMAS["openai-provider-audit"]["properties"][
            "replay_response"
        ]
        self.assertFalse(embedded["additionalProperties"])
        self.assertEqual(
            set(embedded["properties"]),
            set(SCHEMAS["llm-response"]["properties"]),
        )

    def test_contracts_are_standalone_and_encode_strict_cardinality(self) -> None:
        names = (
            "external-decoder-request",
            "external-decoder-judgment",
            "decoder-truth-label",
            "human-collection",
            "heldout-paraphrase-case",
            "heldout-paraphrase-evaluation",
            "heldout-paraphrase-criterion",
            "openai-provider-audit",
            "external-decoder-provider-audit",
            "external-decoder-transport-attempt",
            "native-action-provider-audit",
            "native-action-transport-attempt",
        )
        for name in names:
            with self.subTest(schema=name):
                schema = SCHEMAS[name]
                self.assertIn("/2020-12/", schema["$schema"])
                self.assertEqual(
                    schema["$id"],
                    f"urn:cape-loop:schema:{name}:v1",
                )
                self.assertNotIn("$ref", str(schema))

        probability_rows = SCHEMAS["external-decoder-judgment"][
            "properties"
        ]["probabilities"]
        self.assertEqual(probability_rows["minItems"], 3)
        self.assertEqual(probability_rows["maxItems"], 3)
        self.assertFalse(probability_rows["items"]["additionalProperties"])

        theta = SCHEMAS["decoder-truth-label"]["properties"]["theta"]
        self.assertEqual(theta["minItems"], 3)
        self.assertEqual(theta["maxItems"], 3)

        criterion = SCHEMAS["heldout-paraphrase-criterion"]["properties"]
        self.assertEqual(criterion["verified"]["type"], ["boolean", "null"])
        self.assertEqual(
            criterion["gate_1_argument"]["type"], ["boolean", "null"]
        )


if __name__ == "__main__":
    unittest.main()
