from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import io
import json
import unittest

from cape_loop.beliefs import PreferenceBelief
from cape_loop.artifacts import RunArtifacts
from cape_loop.calibration import TemperatureCalibration
from cape_loop.cli import main as cli_main
from cape_loop.config import (
    AppConfig,
    ConfigError,
    ExperimentSection,
    InferenceSection,
    LLMSection,
    RunSection,
)
from cape_loop.decoder_study import (
    ExternalDecoderRequest,
    build_blinded_native_decoder_request,
    external_decoder_judgment_from_response,
    external_decoder_llm_request,
)
from cape_loop.domains import TRAVEL
from cape_loop.experiments import build_terminal_battery, run_trajectory
from cape_loop.llm_exchange import (
    ATTRIBUTES,
    VALUES,
    LLMRequest,
    LLMResponse,
    ReplayProvider,
    TemperatureCalibratedProvider,
    write_requests,
)
from cape_loop.llm_outcomes import (
    cached_outcome_manifest,
    score_cached_raw_calibrated_terminal,
)
from cape_loop.native import NativeMemoryState
from cape_loop.policies import BalancedPolicy
from cape_loop.runner import _archive_failed_live_attempt, run_experiment
from cape_loop.runner import (
    _live_completion_provider,
    _prepare_llm_execution,
    _prepare_study,
    _run_b,
    _run_c,
)
from cape_loop.schemas import LatentUser, Susceptibility
from cape_loop.updaters import LLMReplayUpdater, UpdateViewKind


class LiveConfigurationTests(unittest.TestCase):
    def test_openai_section_is_strict(self) -> None:
        with self.assertRaises(ConfigError):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "llm": {"mode": "unknown"},
                }
            )
        with self.assertRaises(ConfigError):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "llm": {"base_url": "http://example.test"},
                }
            )
        with self.assertRaises(ConfigError):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "llm": {
                        "base_url": "https://user@api.openai.com",
                    },
                }
            )
        with self.assertRaises(ConfigError):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "llm": {"base_url": "https://proxy.example.test"},
                }
            )
        with self.assertRaisesRegex(ConfigError, "dedicated credential"):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "llm": {
                        "base_url": "https://proxy.example.test",
                        "allow_custom_base_url": True,
                    },
                }
            )
        configured = AppConfig.parse(
            {
                "schema_version": 1,
                "llm": {
                    "base_url": "https://proxy.example.test",
                    "allow_custom_base_url": True,
                    "api_key_env": "CAPE_LOOP_PROXY_KEY",
                },
            }
        )
        self.assertEqual(configured.llm.api_key_env, "CAPE_LOOP_PROXY_KEY")

    def test_openai_section_rejects_other_provider_credentials(self) -> None:
        for reserved_key in (
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
        ):
            with self.subTest(reserved_key=reserved_key):
                with self.assertRaisesRegex(
                    ConfigError,
                    "reserved for a different provider",
                ):
                    AppConfig.parse(
                        {
                            "schema_version": 1,
                            "llm": {
                                "mode": "openai",
                                "api_key_env": reserved_key,
                            },
                        }
                    )

    def test_live_llm_run_cannot_claim_deterministic_generation(self) -> None:
        with self.assertRaisesRegex(
            ConfigError,
            "run.deterministic = false",
        ):
            AppConfig(
                experiment=ExperimentSection(
                    updaters=("llm_full_context",),
                ),
                llm=LLMSection(mode="openai"),
            ).validated()

    def test_temperature_calibration_must_fit_development_population(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ConfigError,
            "calibration_users exceeds the generated development population",
        ):
            AppConfig(
                run=RunSection(deterministic=False),
                experiment=ExperimentSection(
                    updaters=("llm_full_context",),
                    users=1,
                ),
                llm=LLMSection(
                    mode="openai",
                    calibration="temperature",
                    calibration_users=9,
                ),
            ).validated()

    def test_programmatic_adaptive_config_cannot_bypass_key_isolation(
        self,
    ) -> None:
        config = AppConfig(
            run=RunSection(deterministic=False),
            experiment=ExperimentSection(
                updaters=("llm_full_context",),
            ),
            llm=LLMSection(
                mode="openai",
                api_key_env="OPENROUTER_API_KEY",
            ),
        )
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "reserved for a different provider",
            ):
                _live_completion_provider(
                    config,
                    destination=Path(directory) / "adaptive-run",
                    execute_live=True,
                )

    def test_openai_mode_needs_explicit_runtime_authorization(self) -> None:
        with TemporaryDirectory() as directory:
            config = AppConfig(
                run=RunSection(
                    name="no-live-side-effect",
                    output_root=directory,
                    deterministic=False,
                ),
                experiment=ExperimentSection(
                    kind="provenance_audit",
                    domains=("travel",),
                    mechanisms=("balanced", "restricted"),
                    response_modes=("naturally_sampled",),
                    policies=("balanced",),
                    updaters=("fitted_action_aware", "llm_full_context"),
                    users=1,
                    trajectories_per_cell=1,
                    turns=1,
                    bootstrap_replicates=0,
                ),
                inference=InferenceSection(
                    training_interactions=8,
                    fit_steps=2,
                    learning_rate=0.02,
                    l2=0.0,
                    calibration="none",
                ),
                llm=LLMSection(
                    mode="openai",
                    max_retries=0,
                    max_requests=900,
                    max_total_tokens=6_000_000,
                ),
            )
            with self.assertRaisesRegex(ValueError, "--execute-live"):
                run_experiment(config)
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_static_plan_reads_no_key_and_reports_budget(self) -> None:
        request = LLMRequest.build(
            request_id="plan-one",
            updater_id="llm_full_context",
            view="full_context",
            prior={},
            observation={"selected_option": "a"},
            context={"options": ["a", "b"]},
        )
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "requests.jsonl"
            write_requests(request_path, (request,))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = cli_main(
                    [
                        "llm",
                        "plan",
                        str(request_path),
                        "--api-key-env",
                        "ABSENT_CAPE_LOOP_KEY",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["live_execution"])
            self.assertFalse(payload["credential_read"])
            self.assertTrue(payload["within_declared_budget"])
            self.assertEqual(
                payload["request_budget_unit"],
                "physical_http_attempt",
            )
            self.assertEqual(
                payload["theoretical_max_transport_attempts"],
                payload["maximum_attempts_per_request"],
            )

    def test_static_plan_requires_retry_expanded_budget_capacity(self) -> None:
        request = LLMRequest.build(
            request_id="retry-plan-one",
            updater_id="llm_full_context",
            view="full_context",
            prior={},
            observation={"selected_option": "a"},
            context={"options": ["a", "b"]},
        )
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "requests.jsonl"
            write_requests(request_path, (request,))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = cli_main(
                    [
                        "llm",
                        "plan",
                        str(request_path),
                        "--api-key-env",
                        "ABSENT_CAPE_LOOP_KEY",
                        "--max-retries",
                        "2",
                        "--max-requests",
                        "2",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(status, 1)
            self.assertEqual(
                payload["theoretical_max_transport_attempts"],
                3,
            )
            self.assertFalse(payload["within_declared_budget"])

    def test_failed_live_attempt_is_preserved_before_resume(self) -> None:
        with TemporaryDirectory() as directory:
            config = AppConfig(
                run=RunSection(
                    name="failed-live",
                    output_root=directory,
                    deterministic=False,
                ),
                experiment=ExperimentSection(
                    kind="provenance_audit",
                    domains=("travel",),
                    mechanisms=("balanced",),
                    response_modes=("naturally_sampled",),
                    policies=("balanced",),
                    updaters=("llm_full_context",),
                    users=1,
                    trajectories_per_cell=1,
                    turns=1,
                    bootstrap_replicates=0,
                ),
                llm=LLMSection(mode="openai"),
            ).validated()
            run = RunArtifacts.create(config)
            manifest_path = run.path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "failed"
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            archived = _archive_failed_live_attempt(run.path, config)
            self.assertFalse(run.path.exists())
            self.assertTrue(archived.is_dir())
            self.assertIn(".failed-runs", archived.parts)

    def test_llm_calibration_probe_uses_development_only(self) -> None:
        observed_requests: list[LLMRequest] = []

        class UniformProvider:
            def complete(self, request: LLMRequest) -> LLMResponse:
                observed_requests.append(request)
                return LLMResponse.parse(
                    {
                        "schema_version": 1,
                        "request_id": request.request_id,
                        "prompt_sha256": request.prompt_sha256,
                        "model_id": "uniform-fixture",
                        "beliefs": {
                            attribute: {
                                value: 0.25 for value in VALUES
                            }
                            for attribute in ATTRIBUTES
                        },
                    }
                )

        config = AppConfig(
            run=RunSection(
                name="calibration-probe",
                seed=7,
                deterministic=False,
            ),
            experiment=ExperimentSection(
                kind="provenance_audit",
                domains=("travel",),
                mechanisms=("balanced", "restricted"),
                response_modes=("naturally_sampled",),
                policies=("balanced",),
                updaters=("llm_full_context",),
                users=1,
                trajectories_per_cell=1,
                turns=1,
                bootstrap_replicates=0,
            ),
            inference=InferenceSection(
                training_interactions=24,
                fit_steps=10,
                learning_rate=0.03,
                l2=0.001,
                calibration="temperature",
            ),
            llm=LLMSection(
                mode="openai",
                calibration="temperature",
                calibration_users=1,
            ),
        ).validated()
        execution = _prepare_llm_execution(
            config,
            _prepare_study(config),
            raw_provider=UniformProvider(),
        )
        calibration = execution.calibrations["llm_full_context"]
        self.assertEqual(calibration.fitted_splits, ("development",))
        self.assertGreater(calibration.example_count, 0)
        self.assertTrue(execution.development_metrics)
        self.assertTrue(observed_requests)
        contexts = [
            request.payload.get("context", {})
            for request in observed_requests
            if "context" in request.payload
        ]
        self.assertTrue(contexts)
        self.assertTrue(
            all(
                "wording_template" not in context
                for context in contexts
            )
        )
        self.assertTrue(
            all(
                all(
                    option["option_id"].startswith("presented_option_")
                    for option in context["options"]
                )
                for context in contexts
            )
        )
        self.assertTrue(
            all(
                row["split"] == "development"
                for row in execution.development_metrics
            )
        )


class ExternalDecoderAdapterTests(unittest.TestCase):
    def test_native_request_is_blinded_and_provider_response_is_bound(self) -> None:
        state = NativeMemoryState.empty(
            "episodic",
            PreferenceBelief.uniform(),
        )
        request = build_blinded_native_decoder_request(
            state,
            evaluation_split="test",
            assignment_nonce="unit-1",
        )
        reparsed = ExternalDecoderRequest.parse(request.to_dict())
        self.assertEqual(reparsed, request)
        serialized = json.dumps(request.payload)
        self.assertNotIn("memory_kind", serialized)
        self.assertNotIn("updater_id", serialized)
        provider_request = external_decoder_llm_request(
            request,
            decoder_instance_id="decoder-one",
        )
        beliefs = {
            attribute: {value: 0.25 for value in VALUES}
            for attribute in ATTRIBUTES
        }
        provider_response = LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": provider_request.request_id,
                "prompt_sha256": provider_request.prompt_sha256,
                "model_id": "test-decoder",
                "beliefs": beliefs,
            }
        )
        judgment = external_decoder_judgment_from_response(
            request,
            provider_response,
            decoder_instance_id="decoder-one",
            decoder_family_id="family-one",
            source_descriptor="source-one",
        )
        self.assertTrue(judgment.blind_to_system_identity)
        self.assertTrue(judgment.blind_to_latent_truth)

    def test_llm_temperature_wrapper_retains_raw_response(self) -> None:
        request = LLMRequest.build(
            request_id="calibration-one",
            updater_id="llm_full_context",
            view="full_context",
            prior={},
            observation={"selected_option": "a"},
            context={"options": ["a", "b"]},
        )
        raw = LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": request.request_id,
                "prompt_sha256": request.prompt_sha256,
                "model_id": "fixture",
                "beliefs": {
                    attribute: {
                        "-2": 0.7,
                        "-1": 0.1,
                        "+1": 0.1,
                        "+2": 0.1,
                    }
                    for attribute in ATTRIBUTES
                },
            }
        )
        wrapper = TemperatureCalibratedProvider(
            ReplayProvider((raw,)),
            {
                "llm_full_context": TemperatureCalibration(
                    temperature=2.0,
                    fitted_splits=("development",),
                    example_count=12,
                )
            },
        )
        calibrated = wrapper.complete(request)
        self.assertEqual(wrapper.raw_responses, (raw,))
        self.assertLess(
            calibrated.beliefs["attribute_1"]["-2"],
            raw.beliefs["attribute_1"]["-2"],
        )

    def test_cached_terminal_scores_are_paired_but_not_recursive_raw_run(
        self,
    ) -> None:
        class CountingProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, request: LLMRequest) -> LLMResponse:
                self.calls += 1
                return LLMResponse.parse(
                    {
                        "schema_version": 1,
                        "request_id": request.request_id,
                        "prompt_sha256": request.prompt_sha256,
                        "model_id": "counting-fixture",
                        "beliefs": {
                            attribute: {
                                "-2": 0.7,
                                "-1": 0.1,
                                "+1": 0.1,
                                "+2": 0.1,
                            }
                            for attribute in ATTRIBUTES
                        },
                    }
                )

        raw_provider = CountingProvider()
        wrapper = TemperatureCalibratedProvider(
            raw_provider,
            {
                "llm_full_context": TemperatureCalibration(
                    temperature=2.0,
                    fitted_splits=("development",),
                    example_count=12,
                )
            },
        )
        updater = LLMReplayUpdater(
            "llm_full_context",
            UpdateViewKind.FULL_CONTEXT,
            wrapper,
        )
        user = LatentUser(
            "cached-outcome-user",
            (2, -1, 1),
            Susceptibility(0.2, 0.3, 0.4),
        )
        trajectory = run_trajectory(
            user=user,
            domain=TRAVEL,
            policy=BalancedPolicy(),
            updater=updater,
            turns=2,
            seed=11,
        )
        calls_before_scoring = raw_provider.calls
        outcomes = score_cached_raw_calibrated_terminal(
            experiment="B",
            pairing_id=trajectory.trajectory_id,
            split="test",
            regime="closed_loop/balanced/empty",
            updater_id=updater.updater_id,
            active_terminal_belief=trajectory.terminal_belief,
            audit_record=trajectory.audit_record,
            user=user,
            battery=build_terminal_battery(TRAVEL),
            raw_responses=wrapper.raw_responses,
            calibrated_responses=wrapper.calibrated_responses,
        )
        self.assertEqual(raw_provider.calls, calls_before_scoring)
        self.assertEqual(
            {row.calibration_variant for row in outcomes},
            {"raw", "calibrated"},
        )
        self.assertEqual(outcomes[0].request_id, outcomes[1].request_id)
        self.assertTrue(outcomes[0].full_counterfactual_rerun_required)
        payload = outcomes[0].to_dict()
        self.assertEqual(
            payload["estimand_scope"],
            "same-realized-history-terminal-forecast",
        )
        self.assertEqual(payload["provider_calls_added"], 0)
        manifest = cached_outcome_manifest("B", outcomes)
        self.assertEqual(manifest["pair_count"], 1)
        self.assertFalse(manifest["ranking_or_gate_inputs_replaced"])

    def test_b_and_c_write_cached_terminal_outcome_artifacts(self) -> None:
        class ConstantProvider:
            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse.parse(
                    {
                        "schema_version": 1,
                        "request_id": request.request_id,
                        "prompt_sha256": request.prompt_sha256,
                        "model_id": "constant-fixture",
                        "beliefs": {
                            attribute: {
                                "-2": 0.55,
                                "-1": 0.15,
                                "+1": 0.15,
                                "+2": 0.15,
                            }
                            for attribute in ATTRIBUTES
                        },
                    }
                )

        def wrapper() -> TemperatureCalibratedProvider:
            return TemperatureCalibratedProvider(
                ConstantProvider(),
                {
                    "llm_full_context": TemperatureCalibration(
                        temperature=2.0,
                        fitted_splits=("development",),
                        example_count=12,
                    )
                },
            )

        common_inference = InferenceSection(
            training_interactions=24,
            fit_steps=10,
            learning_rate=0.03,
            l2=0.001,
        )
        with TemporaryDirectory() as directory:
            b_config = AppConfig(
                run=RunSection(
                    name="cached-b",
                    seed=19,
                    deterministic=False,
                ),
                experiment=ExperimentSection(
                    kind="closed_loop",
                    domains=("travel",),
                    mechanisms=("ranking", "default", "suggestion"),
                    response_modes=("naturally_sampled",),
                    policies=("balanced", "soft_profile_conditioned"),
                    updaters=("llm_full_context",),
                    users=1,
                    turns=2,
                    bootstrap_replicates=0,
                ),
                inference=common_inference,
                llm=LLMSection(mode="openai"),
            ).validated()
            b_run = RunArtifacts.create(
                b_config,
                root=Path(directory) / "b",
            )
            b_wrapper = wrapper()
            b_summary = _run_b(
                b_config,
                b_run,
                _prepare_study(b_config),
                completion_provider=b_wrapper,
                calibrated_provider=b_wrapper,
            )
            b_manifest = json.loads(
                (
                    b_run.path
                    / "metrics"
                    / "experiment-b-llm-raw-calibrated-terminal-manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                b_manifest["pair_count"],
                b_summary["trajectories"],
            )
            self.assertEqual(
                b_manifest["full_counterfactual_rerun_required_pair_count"],
                b_manifest["pair_count"],
            )

            c_config = AppConfig(
                run=RunSection(
                    name="cached-c",
                    seed=23,
                    deterministic=False,
                ),
                experiment=ExperimentSection(
                    kind="evaluation_validity",
                    domains=("travel",),
                    mechanisms=("ranking", "default", "suggestion"),
                    response_modes=("naturally_sampled",),
                    policies=(
                        "balanced",
                        "fixed_bias",
                        "soft_profile_conditioned",
                    ),
                    updaters=("llm_full_context", "full_context_blind"),
                    users=1,
                    turns=2,
                    bootstrap_replicates=5,
                ),
                inference=common_inference,
                llm=LLMSection(mode="openai"),
            ).validated()
            c_run = RunArtifacts.create(
                c_config,
                root=Path(directory) / "c",
            )
            c_wrapper = wrapper()
            c_summary = _run_c(
                c_config,
                c_run,
                _prepare_study(c_config),
                completion_provider=c_wrapper,
                calibrated_provider=c_wrapper,
            )
            c_manifest = json.loads(
                (
                    c_run.path
                    / "metrics"
                    / "experiment-c-llm-raw-calibrated-terminal-manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                c_manifest["row_count"],
                c_summary["llm_raw_calibrated_terminal_rows"],
            )
            self.assertGreater(c_manifest["pair_count"], 0)
            self.assertFalse(
                c_manifest["ranking_or_gate_inputs_replaced"]
            )


if __name__ == "__main__":
    unittest.main()
