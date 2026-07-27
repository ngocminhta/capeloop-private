from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

from cape_loop.control_study import (
    CLAIM_STATUS,
    CONTROL_IDS,
    ControlLLMRequest,
    build_control_llm_exchange,
    build_experiment_a_control_plan,
    execute_control_llm_exchange,
    read_control_request_bindings,
    run_diagnostic_control_executions,
    validate_control_response_coverage,
    write_control_provider_requests,
    write_control_request_bindings,
)
from cape_loop.cli import main as cli_main
from cape_loop.llm_exchange import (
    ATTRIBUTES,
    VALUES,
    LLMResponse,
    ReplayProvider,
)
from cape_loop.openai_provider import read_requests


def _beliefs(positive_mass: float) -> dict[str, dict[str, float]]:
    negative = (1.0 - positive_mass) / 2.0
    positive = positive_mass / 2.0
    rows = {
        "-2": negative,
        "-1": negative,
        "+1": positive,
        "+2": positive,
    }
    uniform = {value: 0.25 for value in VALUES}
    return {
        ATTRIBUTES[0]: rows,
        ATTRIBUTES[1]: uniform,
        ATTRIBUTES[2]: uniform,
    }


def _responses_for_exchange(exchange) -> tuple[LLMResponse, ...]:
    positive_controls = set(CONTROL_IDS[:3])
    return tuple(
        LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": wrapped.llm_request.request_id,
                "prompt_sha256": wrapped.llm_request.prompt_sha256,
                "model_id": "fixture-model",
                "beliefs": _beliefs(
                    0.85
                    if wrapped.control_id in positive_controls
                    else 0.50
                ),
                "raw_response_sha256": None,
            }
        )
        for wrapped in exchange.requests
    )


class ExperimentAControlPlanTests(unittest.TestCase):
    def test_plan_is_deterministic_bound_and_covers_all_six_controls(self) -> None:
        first = build_experiment_a_control_plan()
        second = build_experiment_a_control_plan()

        self.assertEqual(first, second)
        self.assertEqual(first.plan_sha256, second.plan_sha256)
        self.assertEqual(
            tuple(stimulus.control_id for stimulus in first.stimuli),
            CONTROL_IDS,
        )
        self.assertEqual(first.to_dict()["claim_status"], CLAIM_STATUS)
        self.assertEqual(
            len({stimulus.stimulus_sha256 for stimulus in first.stimuli}),
            6,
        )

        repeated = first.stimulus(
            "positive-repeated-balanced-cross-context"
        )
        self.assertEqual(len(repeated.events), 3)
        self.assertEqual(len({event.scenario_id for event in repeated.events}), 3)
        self.assertEqual(
            len(
                {
                    event.wording_template_id
                    for event in repeated.events
                }
            ),
            3,
        )
        for event in repeated.events:
            target_features = {
                option.features[repeated.target_attribute]
                for option in event.options
            }
            self.assertTrue(min(target_features) < 0 < max(target_features))

        random_control = first.stimulus("negative-random-choice")
        registration = first.randomization_registration
        for event in random_control.events:
            index, digest = registration.draw(
                event_id=event.event_id,
                option_count=len(event.options),
            )
            self.assertEqual(
                event.selected_option_id,
                event.options[index].option_id,
            )
            self.assertEqual(event.randomization_draw_sha256, digest)
            self.assertEqual(
                event.randomization_registration_sha256,
                registration.registration_sha256,
            )

        nondistinguishing = first.stimulus(
            "negative-nondistinguishing-response"
        )
        self.assertEqual(
            len(
                {
                    option.features[nondistinguishing.target_attribute]
                    for option in nondistinguishing.events[0].options
                }
            ),
            1,
        )

    def test_stimulus_tampering_breaks_content_binding(self) -> None:
        plan = build_experiment_a_control_plan()
        stimulus = plan.stimuli[0]
        with self.assertRaises(ValueError):
            replace(
                stimulus,
                minimum_directional_mass_delta=0.25,
            )


class DiagnosticControlExecutionTests(unittest.TestCase):
    def test_reference_and_baseline_are_complete_but_not_empirical(self) -> None:
        plan = build_experiment_a_control_plan()
        reference, baseline = run_diagnostic_control_executions(plan)

        self.assertEqual(reference.required_control_ids, CONTROL_IDS)
        self.assertEqual(baseline.required_control_ids, CONTROL_IDS)
        self.assertEqual(reference.criterion_pass_count, 6)
        self.assertEqual(baseline.criterion_pass_count, 3)
        self.assertEqual(reference.live_evidence_count, 0)
        self.assertEqual(baseline.live_evidence_count, 0)
        self.assertEqual(
            reference.to_dict()["evidence_class"],
            "diagnostic_reference",
        )
        self.assertEqual(
            baseline.to_dict()["evidence_class"],
            "diagnostic_baseline",
        )
        self.assertEqual(reference.to_dict()["claim_status"], CLAIM_STATUS)
        self.assertTrue(
            all(
                outcome.to_dict()["claim_status"] == CLAIM_STATUS
                for outcome in reference.outcomes + baseline.outcomes
            )
        )

        correction = next(
            outcome
            for outcome in reference.outcomes
            if outcome.control_id == "positive-direct-correction"
        )
        assert correction.reference_binding is not None
        self.assertEqual(
            correction.reference_binding["reference_family"],
            "correction_debt_reference_adapter",
        )
        self.assertEqual(
            correction.reference_binding["adapter_id"],
            "reference_log_odds_correction_v1",
        )
        self.assertLess(
            correction.reference_binding["wrong_profile_mass_after"],
            correction.reference_binding["wrong_profile_mass_before"],
        )

        negative = tuple(
            outcome
            for outcome in reference.outcomes
            if outcome.polarity == "negative"
        )
        self.assertTrue(
            all(outcome.directional_mass_delta == 0.0 for outcome in negative)
        )

    def test_report_rejects_incomplete_outcome_coverage(self) -> None:
        plan = build_experiment_a_control_plan()
        reference, _ = run_diagnostic_control_executions(plan)
        with self.assertRaises(ValueError):
            replace(reference, outcomes=reference.outcomes[:-1])


class ControlLLMExchangeTests(unittest.TestCase):
    def test_views_only_receive_semantically_valid_controls(self) -> None:
        plan = build_experiment_a_control_plan()
        response_only = build_control_llm_exchange(
            plan,
            updater_id="llm_response_only_control",
            view="response_only",
        )
        full_context = build_control_llm_exchange(
            plan,
            updater_id="llm_full_context_control",
            view="full_context",
        )
        aware = build_control_llm_exchange(plan)

        self.assertEqual(len(response_only.requests), 3)
        self.assertEqual(len(response_only.omitted_controls), 3)
        self.assertEqual(len(full_context.requests), 5)
        self.assertEqual(len(full_context.omitted_controls), 1)
        self.assertEqual(
            full_context.omitted_controls[0][0],
            "negative-random-choice",
        )
        self.assertEqual(len(aware.requests), 6)
        self.assertFalse(aware.omitted_controls)
        self.assertTrue(
            aware.to_dict()["coverage"]["complete_for_all_six_controls"]
        )

        # Audit labels and expected directions live only in the outer binding;
        # they are not leaked to the model-visible payload.
        for wrapped in aware.requests:
            payload = json.dumps(
                wrapped.llm_request.payload,
                sort_keys=True,
            )
            self.assertNotIn(wrapped.control_id, payload)
            self.assertNotIn('"polarity"', payload)
            self.assertNotIn("expected_diagnostic", payload)

        random_request = next(
            wrapped.llm_request
            for wrapped in aware.requests
            if wrapped.control_id == "negative-random-choice"
        )
        self.assertIn(
            "registered_randomization",
            random_request.payload["provenance"]["event_sequence"][0],
        )

    def test_replay_execution_is_exactly_bound_and_nonclaiming(self) -> None:
        plan = build_experiment_a_control_plan()
        exchange = build_control_llm_exchange(plan)
        responses = _responses_for_exchange(exchange)
        report = execute_control_llm_exchange(
            plan,
            exchange,
            ReplayProvider(responses),
            execution_mode="provider_replay",
            source_descriptor="checksum-verified offline provider fixture",
        )

        self.assertEqual(report.criterion_pass_count, 6)
        self.assertEqual(report.live_evidence_count, 0)
        self.assertEqual(
            report.to_dict()["evidence_class"],
            "external_model_response",
        )
        self.assertEqual(report.to_dict()["claim_status"], CLAIM_STATUS)
        for wrapped, outcome in zip(exchange.requests, report.outcomes):
            self.assertEqual(
                outcome.request_id,
                wrapped.llm_request.request_id,
            )
            self.assertEqual(
                outcome.prompt_sha256,
                wrapped.llm_request.prompt_sha256,
            )
            self.assertEqual(outcome.model_id, "fixture-model")
            self.assertIsNotNone(outcome.response_sha256)
            self.assertFalse(outcome.is_live_evidence)

        with self.assertRaisesRegex(ValueError, "must be labeled"):
            execute_control_llm_exchange(
                plan,
                exchange,
                ReplayProvider(responses),
                execution_mode="provider_live",
                source_descriptor="incorrect live label",
            )

    def test_coverage_rejects_missing_unexpected_and_prompt_mismatch(self) -> None:
        exchange = build_control_llm_exchange(
            build_experiment_a_control_plan()
        )
        responses = _responses_for_exchange(exchange)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_control_response_coverage(exchange, responses[:-1])

        mismatched = replace(
            responses[0],
            prompt_sha256="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "prompt hash mismatch"):
            validate_control_response_coverage(
                exchange,
                (mismatched, *responses[1:]),
            )

    def test_binding_and_generic_provider_jsonl_are_round_trippable(self) -> None:
        exchange = build_control_llm_exchange(
            build_experiment_a_control_plan()
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = root / "control-bindings.jsonl"
            provider_requests = root / "provider-requests.jsonl"
            self.assertEqual(
                write_control_request_bindings(bindings, exchange),
                6,
            )
            self.assertEqual(
                write_control_provider_requests(provider_requests, exchange),
                6,
            )
            loaded_bindings = read_control_request_bindings(bindings)
            loaded_provider = read_requests(provider_requests)
            loaded_binding_bytes = read_control_request_bindings(
                bindings.read_bytes()
            )
            loaded_provider_bytes = read_requests(
                provider_requests.read_bytes()
            )

        self.assertEqual(loaded_bindings, exchange.requests)
        self.assertEqual(loaded_provider, exchange.llm_requests)
        self.assertEqual(loaded_binding_bytes, exchange.requests)
        self.assertEqual(loaded_provider_bytes, exchange.llm_requests)

        raw = exchange.requests[0].to_dict()
        raw["control_id"] = "tampered-control"
        with self.assertRaisesRegex(ValueError, "does not bind"):
            ControlLLMRequest.parse(raw)

    def test_cli_regenerates_bindings_and_atomically_scores_responses(self) -> None:
        exchange = build_control_llm_exchange(
            build_experiment_a_control_plan()
        )
        responses = _responses_for_exchange(exchange)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = root / "bindings.jsonl"
            response_path = root / "responses.jsonl"
            output = root / "analysis.json"
            write_control_request_bindings(bindings, exchange)
            response_path.write_text(
                "".join(
                    json.dumps(
                        response.to_dict(),
                        sort_keys=True,
                    )
                    + "\n"
                    for response in responses
                ),
                encoding="utf-8",
            )
            with redirect_stdout(StringIO()):
                status = cli_main(
                    [
                        "control-study",
                        "analyze",
                        str(bindings),
                        str(response_path),
                        str(output),
                        "--source-descriptor",
                        "reviewed offline fixture",
                    ]
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(status, 0)
            self.assertEqual(payload["claim_status"], CLAIM_STATUS)
            self.assertEqual(payload["request_count"], 6)
            self.assertEqual(
                payload["report"]["criterion_pass_count"],
                6,
            )
            self.assertEqual(
                payload["report"]["execution_mode"],
                "provider_replay",
            )
            self.assertTrue(payload["binding_file_sha256"])
            self.assertTrue(payload["response_file_sha256"])

            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit):
                    cli_main(
                        [
                            "control-study",
                            "analyze",
                            str(bindings),
                            str(response_path),
                            str(output),
                        ]
                    )

    def test_cli_rejects_response_mutation_before_publication(self) -> None:
        exchange = build_control_llm_exchange(
            build_experiment_a_control_plan()
        )
        responses = _responses_for_exchange(exchange)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = root / "bindings.jsonl"
            response_path = root / "responses.jsonl"
            output = root / "analysis.json"
            write_control_request_bindings(bindings, exchange)
            response_path.write_text(
                "".join(
                    json.dumps(response.to_dict(), sort_keys=True) + "\n"
                    for response in responses
                ),
                encoding="utf-8",
            )

            def mutate_after_analysis(*args, **kwargs):
                report = execute_control_llm_exchange(*args, **kwargs)
                response_path.write_bytes(
                    response_path.read_bytes() + b"\n"
                )
                return report

            with (
                patch(
                    "cape_loop.cli.execute_control_llm_exchange",
                    side_effect=mutate_after_analysis,
                ),
                redirect_stderr(StringIO()),
                redirect_stdout(StringIO()),
                self.assertRaises(SystemExit),
            ):
                cli_main(
                    [
                        "control-study",
                        "analyze",
                        str(bindings),
                        str(response_path),
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
