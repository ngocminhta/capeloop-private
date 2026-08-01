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
from cape_loop.population import initial_profile_belief
from cape_loop.domains import TRAVEL
from cape_loop.policies import BalancedPolicy, SoftProfileConditionedPolicy
from cape_loop.response import RandomUtilityModel, RuleBasedResponseModel
from cape_loop.runner import (
    _profile_conditioning_manipulation,
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
    sensitivity_breadth_coverage,
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
        sensitivity_design: str = "cartesian",
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
                design=sensitivity_design,
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

    def test_profile_conditioning_manipulation_distinguishes_control_and_dose(
        self,
    ) -> None:
        negative_control = _profile_conditioning_manipulation(
            conditioning_strength=0.0,
            visible_action_divergence_rate=0.0,
            treatment_exposure_rate=0.0,
        )
        self.assertEqual(
            negative_control["profile_conditioning_manipulation_status"],
            "negative_control_passed",
        )
        self.assertTrue(
            negative_control[
                "profile_conditioning_manipulation_check_passed"
            ]
        )
        self.assertEqual(
            negative_control[
                "phase_profile_conditioning_manipulation_gate"
            ],
            0.0,
        )

        failed_dose = _profile_conditioning_manipulation(
            conditioning_strength=0.33,
            visible_action_divergence_rate=0.0,
            treatment_exposure_rate=0.25,
        )
        self.assertEqual(
            failed_dose["profile_conditioning_manipulation_status"],
            "active_dose_failed_zero_visible_divergence",
        )
        self.assertFalse(
            failed_dose[
                "profile_conditioning_manipulation_check_passed"
            ]
        )
        self.assertEqual(
            failed_dose[
                "phase_profile_conditioning_manipulation_gate"
            ],
            0.0,
        )

        activated_dose = _profile_conditioning_manipulation(
            conditioning_strength=0.33,
            visible_action_divergence_rate=0.25,
            treatment_exposure_rate=0.25,
        )
        self.assertEqual(
            activated_dose["profile_conditioning_manipulation_status"],
            "active_dose_activated",
        )
        self.assertTrue(
            activated_dose[
                "profile_conditioning_manipulation_check_passed"
            ]
        )
        self.assertEqual(
            activated_dose[
                "phase_profile_conditioning_manipulation_gate"
            ],
            1.0,
        )
        insufficient_coverage = _profile_conditioning_manipulation(
            conditioning_strength=0.33,
            visible_action_divergence_rate=0.25,
            treatment_exposure_rate=0.25,
            prospective_coverage_passed=False,
        )
        self.assertEqual(
            insufficient_coverage[
                "profile_conditioning_manipulation_status"
            ],
            "active_dose_failed_informative_strata_coverage",
        )
        self.assertFalse(
            insufficient_coverage[
                "profile_conditioning_manipulation_check_passed"
            ]
        )

    def test_policy_conditioning_strength_changes_visible_actions(self) -> None:
        belief = initial_profile_belief(
            (-2, -1, 1),
            "incorrect",
            profile_strength=0.8,
        )
        balanced = BalancedPolicy()
        applied_by_strength: dict[float, set[int]] = {}
        for strength in (0.0, 0.33, 0.67, 1.0):
            policy = SoftProfileConditionedPolicy(
                conditioning_strength=strength
            )
            applied_by_strength[strength] = {
                turn
                for turn in range(9)
                if policy.action(
                    TRAVEL,
                    belief,
                    turn=turn,
                    master_seed=7,
                    trajectory_id="policy-dose",
                ).provenance.profile_conditioned
            }
        self.assertEqual(applied_by_strength[0.0], set())
        self.assertTrue(
            applied_by_strength[0.0]
            <= applied_by_strength[0.33]
            <= applied_by_strength[0.67]
            <= applied_by_strength[1.0]
        )
        for turn in range(9):
            neutral = SoftProfileConditionedPolicy(
                conditioning_strength=0.0
            ).action(
                TRAVEL,
                belief,
                turn=turn,
                master_seed=7,
                trajectory_id="policy-dose",
            )
            control = balanced.action(
                TRAVEL,
                belief,
                turn=turn,
                master_seed=7,
                trajectory_id="policy-dose",
            )
            self.assertEqual(neutral.signature(), control.signature())
        full_action = SoftProfileConditionedPolicy(
            conditioning_strength=1.0
        ).action(
            TRAVEL,
            belief,
            turn=0,
            master_seed=7,
            trajectory_id="policy-dose",
        )
        intermediate_action = SoftProfileConditionedPolicy(
            conditioning_strength=0.67
        ).action(
            TRAVEL,
            belief,
            turn=0,
            master_seed=7,
            trajectory_id="policy-dose",
        )
        self.assertEqual(
            full_action.provenance.policy_version,
            "v2-neutral-profile-tie",
        )
        self.assertEqual(
            intermediate_action.provenance.policy_version,
            "v3-conditioning-strength",
        )
        uniform = PreferenceBelief.uniform()
        for turn in range(3):
            tied = SoftProfileConditionedPolicy().action(
                TRAVEL,
                uniform,
                turn=turn,
                master_seed=7,
                trajectory_id="policy-tie",
            )
            tied_control = balanced.action(
                TRAVEL,
                uniform,
                turn=turn,
                master_seed=7,
                trajectory_id="policy-tie",
            )
            self.assertFalse(tied.provenance.profile_conditioned)
            self.assertEqual(tied.signature(), tied_control.signature())

    def test_one_at_a_time_grid_is_baseline_first_and_non_factorial(self) -> None:
        points = sensitivity_grid(
            design="one_at_a_time",
            decision_noise_values=[1.0, 0.7],
            presentation_multipliers=[1.0, 1.5],
            rank_multipliers=[1.0, 0.5],
            default_multipliers=[1.0, 1.5],
            suggestion_multipliers=[1.0, 0.5],
            profile_strength_values=[0.8, 0.65],
            prior_uncertainty_values=[0.0, 0.4],
            trajectory_lengths=[8, 12],
            response_model_families=["random_utility", "rule_based"],
            rule_noise_values=[0.15, 0.25],
        )
        # Baseline + one perturbation on each of eight numeric axes + two
        # rule-based baseline points (one per declared rule-noise value).
        self.assertEqual(len(points), 11)
        self.assertEqual(len({point.point_id for point in points}), 11)

        baseline = points[0]
        numeric_fields = (
            "decision_noise",
            "presentation_multiplier",
            "rank_multiplier",
            "default_multiplier",
            "suggestion_multiplier",
            "profile_strength",
            "prior_uncertainty",
            "trajectory_length",
        )
        random_utility_points = points[:9]
        self.assertTrue(
            all(
                point.response_model_family == "random_utility"
                for point in random_utility_points
            )
        )
        self.assertEqual(
            [
                sum(
                    getattr(point, field) != getattr(baseline, field)
                    for field in numeric_fields
                )
                for point in random_utility_points
            ],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        alternate_family_points = points[9:]
        self.assertEqual(
            [point.rule_noise for point in alternate_family_points],
            [0.15, 0.25],
        )
        for point in alternate_family_points:
            self.assertEqual(
                tuple(getattr(point, field) for field in numeric_fields),
                tuple(getattr(baseline, field) for field in numeric_fields),
            )

    def test_sensitivity_design_rejects_unknown_values(self) -> None:
        with self.assertRaisesRegex(
            ConfigError,
            "sensitivity.design must be",
        ):
            SensitivitySection.parse({"design": "fractional"})
        with self.assertRaisesRegex(
            ValueError,
            "sensitivity design must be",
        ):
            sensitivity_grid(
                design="fractional",
                decision_noise_values=[1.0],
                presentation_multipliers=[1.0],
                profile_strength_values=[0.8],
                trajectory_lengths=[8],
            )

    def test_checked_in_sensitivity_config_loads(self) -> None:
        config = load_config("configs/offline/sensitivity.toml")
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
            load_config("configs/offline/sensitivity.toml").sensitivity.response_model_families,
            ("random_utility", "rule_based"),
        )

    def test_full_sensitivity_oat_config_has_exact_declared_workload(self) -> None:
        config = load_config("configs/offline/sensitivity.toml")
        sensitivity = config.sensitivity
        self.assertEqual(sensitivity.design, "one_at_a_time")
        self.assertEqual(
            (
                sensitivity.decision_noise_values[0],
                sensitivity.presentation_multipliers[0],
                sensitivity.profile_conditioning_strength_values[0],
                sensitivity.rank_multipliers[0],
                sensitivity.default_multipliers[0],
                sensitivity.suggestion_multipliers[0],
                sensitivity.profile_strength_values[0],
                sensitivity.prior_uncertainty_values[0],
                sensitivity.trajectory_lengths[0],
                sensitivity.response_model_families[0],
                sensitivity.rule_noise_values[0],
            ),
            (
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.8,
                0.0,
                8,
                "random_utility",
                0.15,
            ),
        )
        points = sensitivity_grid(
            design=sensitivity.design,
            decision_noise_values=sensitivity.decision_noise_values,
            presentation_multipliers=sensitivity.presentation_multipliers,
            profile_conditioning_strength_values=(
                sensitivity.profile_conditioning_strength_values
            ),
            rank_multipliers=sensitivity.rank_multipliers,
            default_multipliers=sensitivity.default_multipliers,
            suggestion_multipliers=sensitivity.suggestion_multipliers,
            profile_strength_values=sensitivity.profile_strength_values,
            prior_uncertainty_values=sensitivity.prior_uncertainty_values,
            trajectory_lengths=sensitivity.trajectory_lengths,
            response_model_families=sensitivity.response_model_families,
            rule_noise_values=sensitivity.rule_noise_values,
        )
        self.assertEqual(len(points), 22)
        self.assertEqual(sum(point.trajectory_length for point in points), 176)

        # Each point runs one incorrect-profile condition across the declared
        # domain × user × replicate × policy × updater cells.
        cells_per_point = (
            len(config.experiment.domains)
            * config.experiment.users
            * config.experiment.trajectories_per_cell
            * len(config.experiment.policies)
            * len(config.experiment.updaters)
        )
        self.assertEqual(cells_per_point, 1_536)
        self.assertEqual(len(points) * cells_per_point, 33_792)
        self.assertEqual(
            sum(point.trajectory_length for point in points)
            * cells_per_point,
            270_336,
        )

    def test_one_at_a_time_grid_varies_one_axis_or_family(self) -> None:
        points = sensitivity_grid(
            design="one_at_a_time",
            decision_noise_values=[1.0, 0.6],
            presentation_multipliers=[1.0, 0.5],
            rank_multipliers=[1.0, 0.5],
            default_multipliers=[1.0, 0.5],
            suggestion_multipliers=[1.0, 0.5],
            profile_strength_values=[0.8, 0.65],
            prior_uncertainty_values=[0.0, 0.35],
            trajectory_lengths=[3, 6],
            response_model_families=["random_utility", "rule_based"],
            rule_noise_values=[0.15, 0.30],
        )
        self.assertEqual(len(points), 11)
        self.assertEqual(
            sum(point.trajectory_length for point in points),
            36,
        )
        self.assertEqual(
            sum(
                point.response_model_family == "random_utility"
                for point in points
            ),
            9,
        )
        self.assertEqual(
            {
                point.rule_noise
                for point in points
                if point.response_model_family == "rule_based"
            },
            {0.15, 0.30},
        )
        self.assertEqual(
            len({point.point_id for point in points}),
            len(points),
        )

    def test_gate6_breadth_covers_presentation_and_conditional_rule_noise(
        self,
    ) -> None:
        points = sensitivity_grid(
            design="one_at_a_time",
            decision_noise_values=[1.0, 0.7],
            presentation_multipliers=[1.0, 0.5],
            profile_conditioning_strength_values=[1.0, 0.0, 0.5],
            rank_multipliers=[1.0, 0.5],
            default_multipliers=[1.0, 0.5],
            suggestion_multipliers=[1.0, 0.5],
            profile_strength_values=[0.8, 0.65],
            prior_uncertainty_values=[0.0, 0.4],
            trajectory_lengths=[8, 4],
            response_model_families=["random_utility", "rule_based"],
            rule_noise_values=[0.15, 0.25],
        )
        passing = [
            {
                **point.to_dict(),
                "operational_joint_region": True,
            }
            for point in points
        ]
        levels, survival, passed = sensitivity_breadth_coverage(
            points,
            passing,
        )
        self.assertTrue(passed)
        self.assertEqual(levels["presentation_multiplier"], [0.5, 1.0])
        self.assertNotIn("profile_conditioning_strength", levels)
        self.assertEqual(levels["rule_noise"], [0.15, 0.25])
        self.assertTrue(all(survival["presentation_multiplier"].values()))
        self.assertTrue(all(survival["rule_noise"].values()))

        without_presentation_perturbation = [
            row
            for row in passing
            if row["presentation_multiplier"] != 0.5
        ]
        _, presentation_survival, presentation_passed = (
            sensitivity_breadth_coverage(
                points,
                without_presentation_perturbation,
            )
        )
        self.assertFalse(presentation_passed)
        self.assertFalse(presentation_survival["presentation_multiplier"]["0.5"])

        without_rule_noise_perturbation = [
            row
            for row in passing
            if not (
                row["response_model_family"] == "rule_based"
                and row["rule_noise"] == 0.25
            )
        ]
        # A malformed non-rule row carrying the same number must not satisfy
        # this conditional axis.
        without_rule_noise_perturbation.append(
            {
                **points[0].to_dict(),
                "rule_noise": 0.25,
                "operational_joint_region": True,
            }
        )
        _, rule_survival, rule_passed = sensitivity_breadth_coverage(
            points,
            without_rule_noise_perturbation,
        )
        self.assertFalse(rule_passed)
        self.assertFalse(rule_survival["rule_noise"]["0.25"])

    def test_sensitivity_design_is_strict(self) -> None:
        with self.assertRaisesRegex(ConfigError, "sensitivity.design"):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "sensitivity": {"design": "fractional"},
                }
            )
        with self.assertRaisesRegex(ValueError, "sensitivity design"):
            sensitivity_grid(
                design="fractional",
                decision_noise_values=[1.0],
                presentation_multipliers=[1.0],
                profile_strength_values=[0.8],
                trajectory_lengths=[3],
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
            config = self._llm_config(
                directory,
                mode="openai",
                sensitivity_design="one_at_a_time",
            )
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
            conversation_rows = [
                json.loads(line)
                for line in (
                    run.path / "conversations" / "sensitivity.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                summary["conversation_log_artifact"],
                "conversations/sensitivity.jsonl",
            )
            self.assertEqual(
                summary["conversation_log_markdown_artifact"],
                "conversations/sensitivity.md",
            )
            self.assertEqual(
                len(conversation_rows),
                summary["conversation_record_count"],
            )
            self.assertEqual(
                sum(len(row["dialogue"]) for row in conversation_rows),
                summary["conversation_turn_count"],
            )
            self.assertEqual(
                sum(len(row["outcomes"]) for row in conversation_rows),
                summary["conversation_outcome_count"],
            )
            self.assertTrue(
                all(
                    "sensitivity_point_id" in row["conditions"]
                    and "point_id" not in row["conditions"]
                    for row in conversation_rows
                )
            )
            self.assertEqual(
                {
                    model_id
                    for row in conversation_rows
                    for outcome in row["outcomes"]
                    for model_id in outcome["model_ids"]
                },
                {"deterministic-uniform-fixture"},
            )
            readable_conversations = (
                run.path / "conversations" / "sensitivity.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Scenario presenter (assistant)", readable_conversations)
            self.assertIn(
                "profile error after this turn",
                readable_conversations.lower(),
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
            phase_specification = json.loads(
                (
                    run.path
                    / "metrics"
                    / "sensitivity-phase-specification.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                phase_specification["design"],
                "one_at_a_time",
            )
            self.assertFalse(
                phase_specification["interaction_effects_estimable"]
            )
            self.assertEqual(phase_specification["declared_points"], 1)
            criterion_ids = {
                row["criterion_id"]
                for row in phase_specification["criteria"]
            }
            self.assertIn(
                "visible-profile-conditioning-activated",
                criterion_ids,
            )
            self.assertNotIn(
                "wrong-profile-self-confirmation",
                criterion_ids,
            )
            strict_secondary = next(
                row
                for row in phase_specification["secondary_endpoints"]
                if row["endpoint_id"]
                == "strict-wrong-profile-self-confirmation"
            )
            self.assertEqual(
                strict_secondary["metric"],
                "phase_self_confirming_profile_rate",
            )
            self.assertFalse(
                strict_secondary["controls_operational_joint_region"]
            )
            grand = json.loads(
                (
                    run.path / "metrics" / "sensitivity-grand.jsonl"
                ).read_text(encoding="utf-8")
            )
            phase = json.loads(
                (
                    run.path
                    / "metrics"
                    / "sensitivity-phase-points.jsonl"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                phase["phase_visible_action_divergence_rate"],
                grand["phase_visible_action_divergence_rate"],
            )
            self.assertEqual(
                phase[
                    "phase_profile_conditioning_treatment_exposure_rate"
                ],
                grand[
                    "phase_profile_conditioning_treatment_exposure_rate"
                ],
            )
            self.assertEqual(
                phase["profile_conditioning_manipulation_status"],
                grand["profile_conditioning_manipulation_status"],
            )
            occupancy_rows = [
                json.loads(line)
                for line in (
                    run.path
                    / "metrics"
                    / "sensitivity-prospective-strata-occupancy.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                len(occupancy_rows),
                summary["prospective_strata_occupancy_rows"],
            )
            self.assertEqual(
                summary["prospective_strata_occupancy_artifact"],
                "metrics/sensitivity-prospective-strata-occupancy.jsonl",
            )
            self.assertEqual(
                occupancy_rows[0]["occupancy"][
                    "strata_assignment_timing"
                ],
                "before_natural_response",
            )
            decomposition_rows = [
                json.loads(line)
                for line in (
                    run.path
                    / "metrics"
                    / "sensitivity-decomposition.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            by_updater = {
                row["updater_id"]: row for row in decomposition_rows
            }
            self.assertEqual(
                grand["phase_selection_cost"],
                by_updater["llm_full_context"]["mean_selection_cost"],
            )
            self.assertEqual(
                grand["secondary_fitted_aware_selection_cost"],
                by_updater["fitted_action_aware"]["mean_selection_cost"],
            )
            self.assertIn(
                "phase_balanced_information_gain_deficit",
                phase,
            )
            self.assertIn(
                "phase_balanced_disconfirmation_evidence_deficit_log_odds",
                phase,
            )
            self.assertIn(
                "phase_self_confirming_profile_rate",
                phase,
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
