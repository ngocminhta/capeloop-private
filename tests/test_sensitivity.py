from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cape_loop.artifacts import RunArtifacts, verify_run
from cape_loop.config import (
    AppConfig,
    ArtifactSection,
    ConfigError,
    ExperimentSection,
    InferenceSection,
    LLMSection,
    RunSection,
    SensitivitySection,
    load_config,
)
from cape_loop.llm_exchange import ATTRIBUTES, VALUES, LLMRequest, LLMResponse
from cape_loop.population import add_prior_uncertainty
from cape_loop.response import RandomUtilityModel, RuleBasedResponseModel
from cape_loop.runner import (
    _run_sensitivity,
    _sensitivity_llm_request_preflight,
    run_experiment,
)
from cape_loop.sensitivity import (
    PhaseCriterion,
    classify_phase_point,
    evaluate_grid,
    infer_axis_boundaries,
    response_model_at,
    sensitivity_grid,
)
from cape_loop.beliefs import PreferenceBelief


class SensitivityTests(unittest.TestCase):
    @staticmethod
    def _llm_config(
        output_root: str,
        *,
        mode: str = "replay",
        max_retries: int = 2,
        max_requests: int = 100,
    ) -> AppConfig:
        return AppConfig(
            run=RunSection(
                name="llm-sensitivity-test",
                seed=7,
                output_root=output_root,
                deterministic=mode == "replay",
            ),
            experiment=ExperimentSection(
                kind="sensitivity",
                domains=("travel",),
                mechanisms=("ranking", "default", "suggestion"),
                response_modes=("naturally_sampled",),
                policies=("balanced", "soft_profile_conditioned"),
                updaters=("fitted_action_aware", "llm_full_context"),
                users=1,
                trajectories_per_cell=1,
                turns=1,
                bootstrap_replicates=0,
            ),
            inference=InferenceSection(
                training_interactions=24,
                fit_steps=2,
                learning_rate=0.03,
                l2=0.001,
                calibration="none",
            ),
            sensitivity=SensitivitySection(
                decision_noise_values=(1.0,),
                presentation_multipliers=(1.0,),
                profile_strength_values=(0.8,),
                trajectory_lengths=(3,),
            ),
            llm=LLMSection(
                mode=mode,
                responses_file=(
                    "unused-replay.jsonl" if mode == "replay" else ""
                ),
                calibration="none",
                max_retries=max_retries,
                max_requests=max_requests,
            ),
            artifacts=ArtifactSection(
                retain_events=True,
                retain_prompts=True,
                checksum_manifest=True,
            ),
        ).validated()

    def test_grid_is_explicit_and_model_scales_noise(self) -> None:
        points = sensitivity_grid(
            decision_noise_values=[0.5, 1.0],
            presentation_multipliers=[1.0],
            profile_strength_values=[0.8],
            trajectory_lengths=[4, 8],
        )
        self.assertEqual(len(points), 4)
        low_noise = response_model_at(
            points[0],
            beta=1.0,
            rank_scale=0.3,
            default_scale=0.8,
            suggestion_scale=0.6,
        )
        self.assertGreater(low_noise.beta, 1.0)
        rows = evaluate_grid(points, lambda point: {"metric": point.decision_noise})
        self.assertEqual(len(rows), len(points))

    def test_checked_in_sensitivity_config_loads(self) -> None:
        config = load_config("configs/sensitivity.toml")
        self.assertEqual(config.experiment.kind, "sensitivity")
        self.assertEqual(len(config.sensitivity.decision_noise_values), 3)

    def test_full_grid_has_independent_axes_and_alternative_family(self) -> None:
        points = sensitivity_grid(
            decision_noise_values=[1.0],
            presentation_multipliers=[1.0],
            rank_multipliers=[0.5, 1.5],
            default_multipliers=[1.0],
            suggestion_multipliers=[1.0],
            profile_strength_values=[0.8],
            prior_uncertainty_values=[0.0, 0.4],
            trajectory_lengths=[4],
            response_model_families=["random_utility", "rule_based"],
            rule_noise_values=[0.1, 0.2],
        )
        # Two rank levels × two prior levels × (one logit + two rule points).
        self.assertEqual(len(points), 12)
        models = [
            response_model_at(
                point,
                beta=1.0,
                rank_scale=0.3,
                default_scale=0.8,
                suggestion_scale=0.6,
            )
            for point in points
        ]
        self.assertTrue(any(isinstance(model, RandomUtilityModel) for model in models))
        self.assertTrue(any(isinstance(model, RuleBasedResponseModel) for model in models))
        self.assertEqual(
            load_config("configs/sensitivity_full.toml").sensitivity.response_model_families,
            ("random_utility", "rule_based"),
        )

    def test_prior_uncertainty_and_phase_boundaries_are_explicit(self) -> None:
        point = PreferenceBelief.point_mass((2, 2, 2))
        mixed = add_prior_uncertainty(point, 0.5)
        self.assertLess(mixed.probability((2, 2, 2)), 1.0)
        criteria = (
            PhaseCriterion("selection", "selection_cost", "gt", 0.0),
            PhaseCriterion("calibration", "ece", "le", 0.1),
        )
        rows = [
            {
                "point_id": "a",
                "decision_noise": 0.5,
                "profile_strength": 0.8,
                "selection_cost": 0.2,
                "ece": 0.05,
            },
            {
                "point_id": "b",
                "decision_noise": 1.0,
                "profile_strength": 0.8,
                "selection_cost": -0.1,
                "ece": 0.05,
            },
        ]
        self.assertTrue(classify_phase_point(rows[0], criteria)["joint_region"])
        boundaries = infer_axis_boundaries(
            rows,
            criteria,
            axis="decision_noise",
        )
        self.assertEqual(boundaries[0]["passing_coordinates"], [0.5])

    def test_llm_sensitivity_contract_and_retry_expanded_preflight(self) -> None:
        with TemporaryDirectory() as directory:
            replay = self._llm_config(directory)
            replay_preflight = _sensitivity_llm_request_preflight(replay)
            assert replay_preflight is not None
            self.assertEqual(
                replay_preflight["logical_completion_upper_bound"],
                6,
            )
            self.assertIsNone(
                replay_preflight["physical_http_attempt_upper_bound"]
            )

            live = self._llm_config(
                directory,
                mode="openai",
                max_retries=2,
                max_requests=17,
            )
            live_preflight = _sensitivity_llm_request_preflight(live)
            assert live_preflight is not None
            self.assertEqual(
                live_preflight["physical_http_attempt_upper_bound"],
                18,
            )
            self.assertFalse(live_preflight["within_request_ceiling"])
            with self.assertRaisesRegex(
                ValueError,
                "18 physical HTTP attempts",
            ):
                run_experiment(live, execute_live=True)
            self.assertEqual(tuple(Path(directory).iterdir()), ())

            with self.assertRaisesRegex(
                ConfigError,
                "llm.calibration = 'none'",
            ):
                AppConfig(
                    **{
                        **{
                            field: getattr(replay, field)
                            for field in (
                                "schema_version",
                                "run",
                                "experiment",
                                "response_model",
                                "inference",
                                "thresholds",
                                "sensitivity",
                                "artifacts",
                            )
                        },
                        "llm": LLMSection(
                            mode="replay",
                            responses_file="unused.jsonl",
                            calibration="temperature",
                        ),
                    }
                ).validated()

    def test_fake_provider_sensitivity_writes_replayable_artifact(self) -> None:
        class UniformProvider:
            def __init__(self) -> None:
                self.audits: list[dict[str, object]] = []
                self.attempts: list[dict[str, object]] = []

            def complete(self, request: LLMRequest) -> LLMResponse:
                ordinal = len(self.audits) + 1
                self.attempts.extend(
                    (
                        {
                            "schema_version": 1,
                            "event": "started",
                            "attempt_ordinal": ordinal,
                            "request_id": request.request_id,
                        },
                        {
                            "schema_version": 1,
                            "event": "settled",
                            "attempt_ordinal": ordinal,
                            "request_id": request.request_id,
                            "outcome": "accepted",
                        },
                    )
                )
                response = LLMResponse.parse(
                    {
                        "schema_version": 1,
                        "request_id": request.request_id,
                        "prompt_sha256": request.prompt_sha256,
                        "model_id": "deterministic-uniform-fixture",
                        "beliefs": {
                            attribute: {
                                value: 0.25 for value in VALUES
                            }
                            for attribute in ATTRIBUTES
                        },
                    }
                )
                self.audits.append(
                    {
                        "schema_version": 1,
                        "request_id": request.request_id,
                        "prompt_sha256": request.prompt_sha256,
                        "model_returned": response.model_id,
                    }
                )
                return response

            @property
            def used_audit_records(self) -> tuple[dict[str, object], ...]:
                return tuple(self.audits)

            @property
            def used_attempt_records(self) -> tuple[dict[str, object], ...]:
                return tuple(self.attempts)

            def to_manifest(self) -> dict[str, object]:
                return {
                    "schema_version": 1,
                    "provider": "deterministic-fixture",
                    "responses_journal": "external-fixture",
                    "audit_journal": "external-fixture",
                    "attempts_journal": "external-fixture",
                }

        with TemporaryDirectory() as directory:
            config = self._llm_config(directory, mode="openai")
            run = RunArtifacts.create(config, root=directory)
            provider = UniformProvider()
            summary = _run_sensitivity(
                config,
                run,
                completion_provider=provider,
                live_provider=provider,  # type: ignore[arg-type]
            )
            run.finalize(summary)

            valid, errors = verify_run(run.path)
            self.assertTrue(valid, errors)
            requests = [
                json.loads(line)
                for line in (
                    run.path / "llm" / "requests.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            responses = [
                json.loads(line)
                for line in (
                    run.path / "llm" / "responses.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(requests)
            self.assertEqual(
                {row["request_id"] for row in requests},
                {row["request_id"] for row in responses},
            )
            self.assertTrue(
                (run.path / "events" / "sensitivity-trajectories.jsonl")
                .is_file()
            )
            self.assertTrue(
                (run.path / "llm" / "provider-audit.jsonl").is_file()
            )
            self.assertTrue(
                (run.path / "llm" / "transport-attempts.jsonl").is_file()
            )
            provider_manifest = json.loads(
                (
                    run.path / "llm" / "provider-manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                provider_manifest["transport_attempt_event_count"],
                len(provider.used_attempt_records),
            )
            self.assertFalse(
                provider_manifest["credentials_retained"]
            )
            grand = json.loads(
                (
                    run.path / "metrics" / "sensitivity-grand.jsonl"
                ).read_text(encoding="utf-8")
            )
            opportunities = grand[
                "phase_profile_consistent_suggestion_opportunities"
            ]
            rejections = grand[
                "phase_profile_consistent_suggestion_rejections"
            ]
            self.assertLessEqual(rejections, opportunities)
            if opportunities:
                self.assertEqual(
                    grand[
                        "phase_profile_consistent_suggestion_rejection_rate"
                    ],
                    rejections / opportunities,
                )
            else:
                self.assertIsNone(
                    grand[
                        "phase_profile_consistent_suggestion_rejection_rate"
                    ]
                )
            gate_report = json.loads(
                (
                    run.path / "metrics" / "gate-report.json"
                ).read_text(encoding="utf-8")
            )
            gate = next(
                row
                for row in gate_report["gates"]
                if row["gate_id"] == "gate-6"
            )
            self.assertEqual(
                [row["criterion_id"] for row in gate["criteria"]],
                [
                    "another-response-model",
                    "broad-simulator-parameters",
                    "both-domains",
                    "multiple-llm-families",
                    "natural-language-paraphrases",
                    "exact-and-fitted-action-aware-references",
                ],
            )
            self.assertIsNone(gate["criteria"][3]["passed"])
            self.assertIsNone(gate["criteria"][4]["passed"])


if __name__ == "__main__":
    unittest.main()
