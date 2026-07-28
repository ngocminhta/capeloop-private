from __future__ import annotations

from pathlib import Path
import unittest

from cape_loop.config import load_config
from cape_loop.experiment_c_review import NATIVE_UPDATER_IDS
from cape_loop.experiments.closed_loop import INITIAL_PROFILE_CONDITIONS
from cape_loop.experiments.evaluation import ALL_REGIMES
from cape_loop.llm_preflight import require_live_llm_budget


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"
OFFLINE_CONFIG_ROOT = CONFIG_ROOT / "offline"
LIVE_CONFIG_ROOT = CONFIG_ROOT / "live"
B_NATIVE_UPDATERS = frozenset(
    {"semantic_memory", "provenance_linked_memory"}
)


class BudgetBoundedPilotConfigTests(unittest.TestCase):
    def test_a_mixed_effects_pair_has_exact_848_attempt_bound(self) -> None:
        designs = []
        for config_path, mode in (
            (LIVE_CONFIG_ROOT / "experiment_a_openai.toml", "openai"),
            (
                LIVE_CONFIG_ROOT / "experiment_a_openrouter.toml",
                "openrouter",
            ),
        ):
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                designs.append(
                    (
                        config.experiment,
                        config.response_model,
                        config.inference,
                        config.thresholds,
                        config.artifacts,
                    )
                )
                self.assertEqual(config.experiment.prior_strengths, (0.0, 0.7))
                self.assertEqual(config.llm.mode, mode)
                preflight = require_live_llm_budget(config)
                assert preflight is not None
                self.assertEqual(
                    preflight["experiment_request_upper_bound"],
                    768,
                )
                self.assertEqual(
                    preflight["calibration_request_count"],
                    48,
                )
                self.assertEqual(
                    preflight["heldout_paraphrase_request_upper_bound"],
                    32,
                )
                self.assertEqual(
                    preflight["physical_http_attempt_upper_bound"],
                    848,
                )
                self.assertEqual(
                    preflight["maximum_output_token_allocation"],
                    1_736_704,
                )
                self.assertEqual(preflight["request_headroom"], 52)
                self.assertTrue(
                    preflight["within_declared_retry_expanded_bounds"]
                )
        self.assertEqual(designs[0], designs[1])

    def test_b_live_configs_freeze_the_same_864_attempt_design(self) -> None:
        designs = []
        for config_path, mode in (
            (
                LIVE_CONFIG_ROOT / "experiment_b_openai.toml",
                "openai",
            ),
            (
                LIVE_CONFIG_ROOT / "experiment_b_openrouter.toml",
                "openrouter",
            ),
        ):
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                designs.append(
                    (
                        config.experiment,
                        config.response_model,
                        config.inference,
                        config.thresholds,
                        config.artifacts,
                    )
                )
                self.assertEqual(config.experiment.kind, "closed_loop")
                self.assertEqual(config.llm.mode, mode)
                self.assertEqual(config.experiment.users, 8)
                self.assertEqual(config.experiment.turns, 3)
                self.assertEqual(
                    config.experiment.policies,
                    ("balanced", "soft_profile_conditioned"),
                )
                self.assertEqual(config.llm.max_retries, 0)
                self.assertEqual(config.llm.max_requests, 900)
                self.assertEqual(config.llm.max_total_tokens, 6_000_000)

                preflight = require_live_llm_budget(config)
                assert preflight is not None
                self.assertEqual(
                    preflight["experiment_request_upper_bound"],
                    768,
                )
                self.assertEqual(
                    preflight["calibration_request_count"],
                    96,
                )
                self.assertEqual(
                    preflight["physical_http_attempt_upper_bound"],
                    864,
                )
                self.assertEqual(preflight["request_headroom"], 36)
                self.assertEqual(
                    preflight["maximum_output_token_allocation"],
                    1_769_472,
                )
                self.assertTrue(
                    preflight["within_declared_retry_expanded_bounds"]
                )
        self.assertEqual(designs[0], designs[1])

    def test_gate4_source_bounds_decoder_and_native_action_packets(self) -> None:
        config = load_config(
            OFFLINE_CONFIG_ROOT / "gate4_source.toml"
        )
        native_count = len(
            set(config.experiment.updaters) & B_NATIVE_UPDATERS
        )
        external_decoder_requests = (
            (config.experiment.users + max(8, config.experiment.users))
            * len(config.experiment.domains)
            * len(INITIAL_PROFILE_CONDITIONS)
            * config.experiment.trajectories_per_cell
            * len(config.experiment.policies)
            * native_count
        )
        eligible_native_actions = (
            config.experiment.users
            * len(config.experiment.domains)
            * config.experiment.trajectories_per_cell
            * native_count
        )
        self.assertEqual(external_decoder_requests, 640)
        self.assertEqual(eligible_native_actions, 80)

    def test_c_external_rescore_packet_fits_direct_sources(self) -> None:
        config = load_config(
            OFFLINE_CONFIG_ROOT / "experiment_c_rescore_source.toml"
        )
        self.assertEqual(config.experiment.kind, "evaluation_validity")
        self.assertEqual(config.experiment.users, 10)
        self.assertEqual(config.experiment.trajectories_per_cell, 1)
        self.assertEqual(config.experiment.turns, 16)
        self.assertEqual(config.experiment.bootstrap_replicates, 2000)
        native_count = len(
            set(config.experiment.updaters) & NATIVE_UPDATER_IDS
        )
        request_count = (
            2
            * config.experiment.users
            * len(config.experiment.domains)
            * config.experiment.trajectories_per_cell
            * len(ALL_REGIMES)
            * native_count
        )
        self.assertEqual(request_count, 360)

    def test_c_live_pair_has_exact_816_attempt_bound(self) -> None:
        designs = []
        for config_path, mode in (
            (LIVE_CONFIG_ROOT / "experiment_c_openai.toml", "openai"),
            (
                LIVE_CONFIG_ROOT / "experiment_c_openrouter.toml",
                "openrouter",
            ),
        ):
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                designs.append(
                    (
                        config.experiment,
                        config.response_model,
                        config.inference,
                        config.thresholds,
                        config.artifacts,
                    )
                )
                self.assertEqual(config.llm.mode, mode)
                preflight = require_live_llm_budget(config)
                assert preflight is not None
                self.assertEqual(
                    preflight["experiment_request_upper_bound"],
                    768,
                )
                self.assertEqual(
                    preflight["calibration_request_count"],
                    48,
                )
                self.assertEqual(
                    preflight["physical_http_attempt_upper_bound"],
                    816,
                )
                self.assertEqual(
                    preflight["maximum_output_token_allocation"],
                    1_671_168,
                )
                self.assertEqual(preflight["request_headroom"], 84)
                self.assertTrue(
                    preflight["within_declared_retry_expanded_bounds"]
                )
        self.assertEqual(designs[0], designs[1])

    def test_gate6_oat_pair_has_exact_576_attempt_bound(self) -> None:
        designs = []
        for config_path, mode in (
            (LIVE_CONFIG_ROOT / "sensitivity_openai.toml", "openai"),
            (
                LIVE_CONFIG_ROOT / "sensitivity_openrouter.toml",
                "openrouter",
            ),
        ):
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                designs.append(
                    (
                        config.experiment,
                        config.response_model,
                        config.inference,
                        config.thresholds,
                        config.sensitivity,
                        config.artifacts,
                    )
                )
                self.assertEqual(config.llm.mode, mode)
                self.assertEqual(config.sensitivity.design, "one_at_a_time")
                preflight = require_live_llm_budget(config)
                assert preflight is not None
                self.assertEqual(preflight["grid_points"], 11)
                self.assertEqual(
                    preflight["sum_trajectory_lengths_over_points"],
                    36,
                )
                self.assertEqual(
                    preflight["physical_http_attempt_upper_bound"],
                    576,
                )
                self.assertEqual(
                    preflight["maximum_output_token_allocation"],
                    1_179_648,
                )
                self.assertEqual(preflight["request_headroom"], 324)
                self.assertTrue(
                    preflight["within_declared_retry_expanded_bounds"]
                )
        self.assertEqual(designs[0], designs[1])

    def test_every_openrouter_gemini_config_pins_minimal_reasoning(
        self,
    ) -> None:
        config_paths = (
            LIVE_CONFIG_ROOT / "experiment_a_openrouter.toml",
            LIVE_CONFIG_ROOT / "experiment_b_openrouter.toml",
            LIVE_CONFIG_ROOT / "experiment_c_openrouter.toml",
            LIVE_CONFIG_ROOT / "sensitivity_openrouter.toml",
        )
        for config_path in config_paths:
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                self.assertEqual(config.llm.mode, "openrouter")
                self.assertEqual(
                    config.llm.model,
                    "google/gemini-3.6-flash",
                )
                self.assertEqual(config.llm.reasoning_effort, "minimal")


if __name__ == "__main__":
    unittest.main()
