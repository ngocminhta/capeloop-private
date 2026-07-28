from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cape_loop.artifacts import verify_run
from cape_loop.config import (
    AppConfig,
    ExperimentSection,
    InferenceSection,
    LLMSection,
    RunSection,
)
from cape_loop.llm_preflight import (
    build_llm_request_preflight,
    require_live_llm_budget,
)
from cape_loop.runner import PreparedLLMExecution, run_experiment


class AdaptiveLLMRequestPreflightTests(unittest.TestCase):
    @staticmethod
    def _config(
        kind: str,
        *,
        output_root: str = "runs",
        max_retries: int,
        max_requests: int,
        max_output_tokens: int = 100,
        max_total_tokens: int = 1_000_000,
        users: int = 2,
    ) -> AppConfig:
        common = {
            "domains": ("travel",),
            "users": users,
            "trajectories_per_cell": 2,
            "turns": 3,
        }
        if kind == "provenance_audit":
            experiment = ExperimentSection(
                kind=kind,
                mechanisms=("balanced", "restricted"),
                response_modes=(
                    "controlled_anchor",
                    "naturally_sampled",
                ),
                prior_strengths=(0.0, 0.5),
                policies=("balanced",),
                updaters=("llm_response_only", "llm_full_context"),
                users=users,
                trajectories_per_cell=1,
                turns=1,
                bootstrap_replicates=0,
                domains=("travel",),
            )
        elif kind == "closed_loop":
            experiment = ExperimentSection(
                kind=kind,
                mechanisms=("ranking", "default", "suggestion"),
                response_modes=("naturally_sampled",),
                prior_strengths=(0.0,),
                policies=("balanced", "soft_profile_conditioned"),
                updaters=(
                    "llm_full_context",
                    "llm_provenance_aware",
                ),
                bootstrap_replicates=0,
                **common,
            )
        elif kind == "evaluation_validity":
            experiment = ExperimentSection(
                kind=kind,
                mechanisms=("ranking", "default", "suggestion"),
                response_modes=("naturally_sampled",),
                prior_strengths=(0.0,),
                policies=(
                    "balanced",
                    "fixed_bias",
                    "soft_profile_conditioned",
                ),
                updaters=(
                    "llm_response_only",
                    "llm_full_context",
                ),
                bootstrap_replicates=5,
                **common,
            )
        else:
            raise AssertionError(f"unsupported test kind: {kind}")
        return AppConfig(
            run=RunSection(
                name=f"preflight-{kind}",
                seed=11,
                output_root=output_root,
                deterministic=False,
            ),
            experiment=experiment,
            inference=InferenceSection(
                training_interactions=24,
                fit_steps=2,
                learning_rate=0.03,
                l2=0.001,
                calibration="none",
            ),
            llm=LLMSection(
                mode="openai",
                calibration="temperature",
                calibration_users=1,
                max_retries=max_retries,
                max_output_tokens=max_output_tokens,
                max_requests=max_requests,
                max_total_tokens=max_total_tokens,
            ),
        ).validated()

    def test_experiment_a_counts_calibration_and_heldout_paraphrases(
        self,
    ) -> None:
        preflight = build_llm_request_preflight(
            self._config(
                "provenance_audit",
                max_retries=2,
                max_requests=744,
            )
        )
        assert preflight is not None
        # Main: 2 users × 1 domain × 3 attributes × 2 directions ×
        # 2 priors × 2 mechanisms × 2 modes × 2 LLM updaters.
        self.assertEqual(preflight["experiment_request_upper_bound"], 192)
        # Development calibration: 1 user × 1 domain × 3 attributes ×
        # 2 directions × 4 mechanisms × 2 LLM updaters.
        self.assertEqual(preflight["calibration_request_count"], 48)
        # Two source trials per domain/mechanism × two test templates. Only
        # llm_full_context participates in this held-out surface check.
        self.assertEqual(
            preflight["heldout_paraphrase_request_upper_bound"],
            8,
        )
        self.assertEqual(preflight["logical_completion_upper_bound"], 248)
        self.assertEqual(preflight["retry_expansion_factor"], 3)
        self.assertEqual(
            preflight["physical_http_attempt_upper_bound"],
            744,
        )

    def test_experiment_b_counts_all_four_initial_profile_conditions(
        self,
    ) -> None:
        preflight = build_llm_request_preflight(
            self._config(
                "closed_loop",
                max_retries=1,
                max_requests=480,
            )
        )
        assert preflight is not None
        # Main: 2 users × 1 domain × 4 initial profiles × 2 replicates ×
        # 2 policies × 3 turns × 2 LLM updaters.
        self.assertEqual(preflight["experiment_request_upper_bound"], 192)
        self.assertEqual(preflight["calibration_request_count"], 48)
        self.assertEqual(preflight["logical_completion_upper_bound"], 240)
        self.assertEqual(
            preflight["physical_http_attempt_upper_bound"],
            480,
        )

    def test_experiment_c_counts_both_splits_and_all_three_regimes(
        self,
    ) -> None:
        preflight = build_llm_request_preflight(
            self._config(
                "evaluation_validity",
                max_retries=2,
                max_requests=1_224,
                max_total_tokens=200_000,
            )
        )
        assert preflight is not None
        # Main: (8 development + 2 test users) × 1 domain × 2 replicates ×
        # 3 regimes × 3 turns × 2 LLM updaters.
        self.assertEqual(preflight["experiment_request_upper_bound"], 360)
        self.assertEqual(preflight["calibration_request_count"], 48)
        self.assertEqual(preflight["logical_completion_upper_bound"], 408)
        self.assertEqual(
            preflight["physical_http_attempt_upper_bound"],
            1_224,
        )

    def test_experiment_c_small_test_population_counts_all_development_users(
        self,
    ) -> None:
        preflight = build_llm_request_preflight(
            self._config(
                "evaluation_validity",
                users=1,
                max_retries=0,
                max_requests=372,
            )
        )
        assert preflight is not None
        # Main: (8 development + 1 test user) × 1 domain × 2 replicates ×
        # 3 regimes × 3 turns × 2 LLM updaters.
        self.assertEqual(preflight["experiment_request_upper_bound"], 324)
        self.assertEqual(preflight["calibration_request_count"], 48)
        self.assertEqual(preflight["logical_completion_upper_bound"], 372)
        self.assertEqual(
            preflight["physical_http_attempt_upper_bound"],
            372,
        )

    def test_over_ceiling_b_fails_before_provider_or_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            config = self._config(
                "closed_loop",
                output_root=directory,
                max_retries=0,
                max_requests=239,
            )
            with patch(
                "cape_loop.runner._live_completion_provider",
                side_effect=AssertionError(
                    "provider construction preceded request preflight"
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "240 physical HTTP attempts",
                ):
                    run_experiment(config, execute_live=True)
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_output_allocation_must_fit_before_adaptive_input(self) -> None:
        config = self._config(
            "closed_loop",
            max_retries=0,
            max_requests=240,
            max_output_tokens=100,
            max_total_tokens=23_999,
        )
        preflight = build_llm_request_preflight(config)
        assert preflight is not None
        self.assertEqual(
            preflight["maximum_output_token_allocation"],
            24_000,
        )
        self.assertFalse(preflight["within_output_token_ceiling"])
        with self.assertRaisesRegex(
            ValueError,
            "allocate up to 24000 output tokens",
        ):
            require_live_llm_budget(config)

    def test_successful_run_retains_the_reviewed_preflight(self) -> None:
        with TemporaryDirectory() as directory:
            config = self._config(
                "closed_loop",
                output_root=directory,
                max_retries=0,
                max_requests=240,
                max_output_tokens=100,
                max_total_tokens=1_000_000,
            )
            fake_provider = object()
            fake_execution = PreparedLLMExecution(
                raw_provider=fake_provider,
                active_provider=fake_provider,
                development_registry={},
                calibrations={},
                development_metrics=(),
            )
            with (
                patch(
                    "cape_loop.runner._live_completion_provider",
                    return_value=fake_provider,
                ),
                patch(
                    "cape_loop.runner._prepare_study",
                    return_value=object(),
                ),
                patch("cape_loop.runner._write_prepared"),
                patch(
                    "cape_loop.runner._prepare_llm_execution",
                    return_value=fake_execution,
                ),
                patch("cape_loop.runner._write_llm_calibration"),
                patch(
                    "cape_loop.runner._run_b",
                    return_value={"status": "fixture"},
                ),
            ):
                result = run_experiment(config, execute_live=True)

            run_dir = Path(result["run_dir"])
            retained = json.loads(
                (run_dir / "llm" / "request-preflight.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(retained["experiment_kind"], "closed_loop")
            self.assertEqual(
                retained["logical_completion_upper_bound"],
                240,
            )
            self.assertEqual(
                retained["physical_http_attempt_upper_bound"],
                240,
            )
            self.assertTrue(retained["within_declared_retry_expanded_bounds"])
            valid, errors = verify_run(run_dir)
            self.assertTrue(valid, errors)


if __name__ == "__main__":
    unittest.main()
