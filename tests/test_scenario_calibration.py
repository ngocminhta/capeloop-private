from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from cape_loop.cli import main
from cape_loop.conversation_surfaces import load_conversation_bank
from cape_loop.response import RandomUtilityModel
from cape_loop.scenario_calibration import (
    AUDIT_POLICY,
    AUDIT_SCHEMA_VERSION,
    build_scenario_calibration_audit,
    render_blinded_surface_review_markdown,
    render_scenario_calibration_markdown,
)
from cape_loop.scenarios import ScenarioSpec, load_scenario_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "scenarios" / "scenario-catalog-v1.json"
BANK_PATH = ROOT / "data" / "scenarios" / "conversation-templates-v1.json"


class CanonicalScenarioCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        material = CATALOG_PATH.read_bytes()
        cls.catalog = load_scenario_catalog(
            CATALOG_PATH,
            expected_sha256=sha256(material).hexdigest(),
        ).catalog
        cls.bank = load_conversation_bank(BANK_PATH)
        cls.model = RandomUtilityModel(
            beta=1.0,
            ranking_scale=0.35,
            default_scale=0.75,
            suggestion_scale=0.65,
        )
        cls.audit = build_scenario_calibration_audit(
            cls.catalog,
            cls.bank,
            cls.model,
            susceptibility_levels=(0.15, 0.45, 0.85),
            split="test",
            planned_turns=4,
        )

    def test_audit_is_json_ready_deterministic_and_outcome_free(self) -> None:
        encoded = json.dumps(
            self.audit,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        repeated = build_scenario_calibration_audit(
            self.catalog,
            self.bank,
            self.model,
            susceptibility_levels=(0.85, 0.15, 0.45),
            split="test",
            planned_turns=4,
        )
        self.assertEqual(repeated, self.audit)
        self.assertEqual(
            json.loads(encoded),
            self.audit,
        )
        self.assertEqual(self.audit["audit_policy"], AUDIT_POLICY)
        self.assertFalse(self.audit["outcome_data_used"])
        self.assertFalse(self.audit["review_status_mutated"])

    def test_no_repeat_capacity_uses_ceiling_and_16_turn_reference(self) -> None:
        capacity = self.audit["capacity"]
        self.assertEqual(capacity["planned_turns"], 4)
        self.assertEqual(
            capacity["planned_capacity_basis"],
            "declared_balanced_coverage_at_most_ceil_turns_over_3",
        )
        self.assertEqual(
            capacity["planned_per_cell_no_repeat_required"],
            2,
        )
        self.assertTrue(capacity["planned_all_cells_sufficient"])
        self.assertEqual(capacity["cyclic_reference_turns"], 16)
        self.assertEqual(
            capacity["cyclic_reference_per_cell_no_repeat_required"],
            6,
        )
        self.assertTrue(capacity["cyclic_reference_all_cells_sufficient"])
        self.assertTrue(
            all(cell["available_scenarios"] == 6 for cell in capacity["cells"])
        )

    def test_declared_primary_model_passes_prospective_guardrails(self) -> None:
        calibration = self.audit["probability_calibration"]
        self.assertEqual(calibration["theta_state_count"], 64)
        self.assertEqual(calibration["susceptibility_profile_count"], 27)
        self.assertEqual(calibration["counterbalanced_order_count"], 2)
        self.assertEqual(
            calibration["scenario_anchor_state_count"],
            36 * 2 * 64 * 27,
        )
        self.assertEqual(calibration["scenario_anchor_instance_count"], 72)
        self.assertEqual(calibration["unique_numeric_signature_count"], 72)
        self.assertEqual(calibration["numeric_signature_repetition_factor"], 1.0)
        self.assertEqual(
            calibration["target_half_spans"],
            [0.1, 0.16, 0.24, 0.34, 0.46, 0.56],
        )
        self.assertEqual(
            calibration["guardrail_scope"]["included_scenario_count"],
            30,
        )
        self.assertEqual(
            calibration["guardrail_scope"]["excluded_decisive_control_scenario_count"],
            6,
        )
        self.assertTrue(calibration["all_cells_passed"])

        guardrails = calibration["guardrails"]
        self.assertTrue(guardrails["order_averaged_balanced_probability"]["passed"])
        self.assertEqual(
            guardrails["order_averaged_balanced_probability"]["outside_count"],
            0,
        )
        self.assertTrue(guardrails["order_averaged_restricted_probability"]["passed"])
        self.assertEqual(
            guardrails["order_averaged_restricted_probability"]["outside_count"],
            0,
        )
        for mechanism in ("balanced", "restricted", "default", "suggestion"):
            with self.subTest(physical_mechanism=mechanism):
                physical = guardrails["physical_mechanism_probabilities"][mechanism]
                self.assertTrue(physical["passed"])
                self.assertEqual(
                    physical["anchor_at_or_below_minimum_count"],
                    0,
                )
                self.assertEqual(
                    physical["anchor_at_or_above_complementary_ceiling_count"],
                    0,
                )
                self.assertEqual(
                    physical["evaluated_physical_probability_count"],
                    30 * 2 * 64 * 27 * 2,
                )
        for mechanism in ("ranking", "default", "suggestion"):
            with self.subTest(mechanism=mechanism):
                effect = guardrails["mean_incremental_effects"][mechanism]
                self.assertTrue(effect["passed"])
                self.assertGreaterEqual(effect["observed_mean"], 0.02)
                self.assertLessEqual(effect["observed_mean"], 0.20)

    def test_readiness_tiers_and_review_counts_are_separate(self) -> None:
        readiness = self.audit["readiness"]
        self.assertTrue(readiness["engineering_pilot"]["ready"])
        self.assertFalse(readiness["scientific_pilot"]["ready"])
        self.assertFalse(readiness["paper"]["ready"])

        reviews = self.audit["human_review_counts"]["selected_split"]
        self.assertEqual(reviews["scenario_count"], 36)
        self.assertEqual(
            reviews["surface_human_review"],
            {"not_completed": 36},
        )
        self.assertEqual(
            reviews["scientific_human_review"],
            {"not_completed": 36},
        )
        self.assertEqual(reviews["paper_eligible_count"], 0)
        self.assertNotIn(
            "independent_surface_human_review_passed",
            readiness["scientific_pilot"]["criteria"],
        )
        self.assertFalse(
            readiness["scientific_pilot"]["criteria"][
                "independent_human_review_evidence_bundle_verified"
            ]
        )
        evidence = self.audit["readiness_contract"]["human_review_evidence"]
        self.assertTrue(evidence["verification_supported"])
        self.assertFalse(evidence["verified"])

    def test_restricted_peer_nuisance_design_is_counterbalanced(self) -> None:
        design = self.audit["restricted_peer_nuisance_design"]
        self.assertTrue(design["all_cells_passed"])
        self.assertEqual(design["magnitude"], 0.25)
        for cell in design["cells"]:
            with self.subTest(
                domain=cell["domain"],
                target=cell["target_key"],
            ):
                self.assertEqual(sorted(cell["attribute_counts"].values()), [3, 3])
                self.assertEqual(
                    sorted(cell["direction_counts"].values()),
                    [3, 3],
                )
                joint = tuple(cell["joint_counts"].values())
                self.assertLessEqual(max(joint) - min(joint), 1)
                self.assertTrue(cell["nuisance_joint_combinations_balanced_within_one"])
                self.assertTrue(cell["passed"])

    def test_neutral_conversation_frames_are_counterbalanced(self) -> None:
        design = self.audit["conversation_frame_design"]
        self.assertTrue(design["passed"])
        self.assertTrue(design["source_neutral_heuristic_passed"])
        self.assertEqual(design["assistant_agency_candidates"], [])
        self.assertEqual(design["frame_family_count"], 3)
        self.assertEqual(
            sorted(design["frame_counts"].values()),
            [12, 12, 12],
        )
        for cell in design["cells"]:
            with self.subTest(
                domain=cell["domain"],
                target=cell["target_key"],
            ):
                self.assertEqual(
                    sorted(cell["frame_counts"].values()),
                    [2, 2, 2],
                )
                self.assertTrue(cell["balanced_within_one"])

    def test_canonical_machine_warnings_and_hygiene_are_clean(self) -> None:
        warnings = self.audit["warnings"]
        self.assertFalse(warnings["blocks_machine_readiness"])
        self.assertFalse(warnings["blocks_recorded_scientific_readiness"])
        self.assertTrue(warnings["version_bound_adjudication_mechanism_available"])
        self.assertEqual(warnings["lexical_overlap_candidates"], [])
        self.assertEqual(warnings["rendered_surface_hygiene"], [])
        self.assertEqual(warnings["raw_option_label_word_count_warnings"], [])
        self.assertEqual(
            warnings["warning_count"],
            len(warnings["all"]),
        )
        self.assertTrue(
            all(
                warning["blocks_machine_readiness"] is False
                for warning in warnings["all"]
            )
        )
        self.assertTrue(self.audit["readiness"]["engineering_pilot"]["ready"])
        self.assertTrue(
            self.audit["readiness"]["scientific_pilot"]["criteria"][
                "unresolved_machine_warning_count_is_zero"
            ]
        )

    def test_human_review_packet_renders_every_declared_surface(self) -> None:
        packet = self.audit["human_review_packet"]
        self.assertEqual(len(packet["scenarios"]), 36)
        self.assertTrue(
            all(
                len(scenario["rendered_examples"]) == 6
                for scenario in packet["scenarios"]
            )
        )
        observed = {
            example["review_surface"]
            for scenario in packet["scenarios"]
            for example in scenario["rendered_examples"]
        }
        self.assertEqual(
            observed,
            {
                "balanced",
                "restricted_negative",
                "restricted_positive",
                "default",
                "suggestion",
                "ranking_reversed",
            },
        )
        machine = self.audit["machine_surface_rendering"]
        self.assertTrue(machine["complete"])
        self.assertEqual(machine["expected_surface_count_per_scenario"], 40)
        self.assertEqual(machine["rendered_total_surface_count"], 36 * 40)
        self.assertTrue(machine["hygiene_clean"])
        self.assertEqual(machine["hygiene_warning_count"], 0)
        self.assertTrue(
            all(
                scenario["machine_surface_count"] == 40
                for scenario in packet["scenarios"]
            )
        )

        first = render_scenario_calibration_markdown(self.audit)
        second = render_scenario_calibration_markdown(self.audit)
        self.assertEqual(first, second)
        self.assertIn("# Prospective scenario calibration", first)
        self.assertIn("| Engineering pilot | **yes** |", first)
        self.assertIn("| Recorded scientific pilot | **no** |", first)
        self.assertIn("Restricted-peer nuisance design", first)
        self.assertIn("**Assistant:**", first)
        self.assertIn("**User:**", first)
        self.assertIn("Independent review checklist", first)
        self.assertIn("contains no experiment outcomes", first)
        self.assertIn("detailed researcher packet", first)

    def test_blinded_surface_packet_is_opaque_and_surface_only(self) -> None:
        first = render_blinded_surface_review_markdown(self.audit)
        second = render_blinded_surface_review_markdown(self.audit)
        self.assertEqual(first, second)
        self.assertIn("# Blinded scenario surface-review packet", first)
        self.assertIn("records no review evidence", first)
        self.assertIn("Naturalness (1–5)", first)
        self.assertIn("Neutrality (1–5)", first)

        lines = first.splitlines()
        item_headings = [line for line in lines if line.startswith("## Item ")]
        surface_headings = [line for line in lines if line.startswith("### Surface ")]
        self.assertEqual(len(item_headings), 36)
        self.assertEqual(item_headings[0], "## Item 001")
        self.assertEqual(item_headings[-1], "## Item 036")
        self.assertEqual(len(surface_headings), 36 * 6)
        self.assertTrue(
            all(
                heading in {f"### Surface {index:02d}" for index in range(1, 7)}
                for heading in surface_headings
            )
        )

        for forbidden_label in (
            "- Split:",
            "- Target:",
            "Domain / task:",
            "Restricted peer nuisance:",
            "Features (reviewer aid",
            "| Role |",
            "presentation_mechanism",
            "anchor_direction",
            "nuisance_attribute",
            "nuisance_direction",
        ):
            with self.subTest(forbidden_label=forbidden_label):
                self.assertNotIn(forbidden_label, first)

        headings = "\n".join(line for line in lines if line.startswith("#")).casefold()
        for mechanism in (
            "balanced",
            "restricted",
            "default",
            "suggestion",
            "ranking",
        ):
            with self.subTest(mechanism=mechanism):
                self.assertNotIn(mechanism, headings)

        for scenario in self.audit["human_review_packet"]["scenarios"]:
            self.assertNotIn(scenario["scenario_id"], first)
            for option in scenario["options"]:
                self.assertNotIn(option["option_id"], first)
            for example in scenario["rendered_examples"]:
                self.assertNotIn(
                    f"**{example['review_surface']}**",
                    first,
                )

    def test_catalog_status_strings_cannot_verify_human_evidence(self) -> None:
        reviewed_scenarios = tuple(
            replace(
                scenario,
                status="approved",
                review={
                    **scenario.review,
                    "automated_validation": "passed",
                    "surface_human_review": "passed",
                    "scientific_human_review": "passed",
                    "paper_eligible": True,
                },
            )
            for scenario in self.catalog.scenarios
        )
        reviewed_catalog = replace(
            self.catalog,
            catalog_status="frozen-paper",
            eligibility="paper-eligible",
            scenarios=reviewed_scenarios,
        )
        audit = build_scenario_calibration_audit(
            reviewed_catalog,
            self.bank,
            self.model,
            susceptibility_levels=(0.15, 0.45, 0.85),
            split="test",
            planned_turns=4,
        )
        readiness = audit["readiness"]
        self.assertTrue(readiness["engineering_pilot"]["ready"])
        self.assertFalse(readiness["scientific_pilot"]["ready"])
        self.assertEqual(
            readiness["scientific_pilot"]["blocking_reasons"],
            ["independent_human_review_evidence_bundle_verified"],
        )
        self.assertFalse(readiness["paper"]["ready"])
        self.assertEqual(
            readiness["paper"]["blocking_reasons"],
            ["recorded_scientific_pilot_ready"],
        )

    def test_cli_writes_machine_report_and_review_packet(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "scenarios",
                        "audit",
                        str(ROOT / "configs" / "live" / "experiment_b_openrouter.toml"),
                        str(output),
                        "--split",
                        "test",
                        "--turns",
                        "6",
                    ]
                )
            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertFalse(summary["outcome_data_used"])
            self.assertEqual(summary["planned_turns"], 6)
            self.assertTrue((output / "scenario-audit.json").is_file())
            researcher_packet = output / "scenario-review.md"
            blinded_packet = output / "scenario-surface-review-blinded.md"
            self.assertTrue(researcher_packet.is_file())
            self.assertTrue(blinded_packet.is_file())
            self.assertEqual(
                summary["researcher_review_packet"],
                str(researcher_packet),
            )
            self.assertEqual(
                summary["blinded_surface_review_packet"],
                str(blinded_packet),
            )
            self.assertIn(
                "Independent review checklist",
                researcher_packet.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "# Blinded scenario surface-review packet",
                blinded_packet.read_text(encoding="utf-8"),
            )


class SyntheticScenarioCalibrationBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        material = CATALOG_PATH.read_bytes()
        cls.catalog = load_scenario_catalog(
            CATALOG_PATH,
            expected_sha256=sha256(material).hexdigest(),
        ).catalog
        cls.bank = load_conversation_bank(BANK_PATH)

    @staticmethod
    def _near_copy_surface(
        target: ScenarioSpec,
        source: ScenarioSpec,
        *,
        marker: str,
        task_family: str | None = None,
    ) -> ScenarioSpec:
        return replace(
            target,
            prompt=f"{source.prompt} {marker}",
            task_family=(target.task_family if task_family is None else task_family),
            negative_option=replace(
                target.negative_option,
                label=f"{source.negative_option.label} {marker}",
            ),
            positive_option=replace(
                target.positive_option,
                label=f"{source.positive_option.label} {marker}",
            ),
            negative_same_direction_option=replace(
                target.negative_same_direction_option,
                label=(f"{source.negative_same_direction_option.label} {marker}"),
            ),
            positive_same_direction_option=replace(
                target.positive_same_direction_option,
                label=(f"{source.positive_same_direction_option.label} {marker}"),
            ),
        )

    def test_unknown_adaptive_capacity_is_conservative_and_domain_scoped(
        self,
    ) -> None:
        audit = build_scenario_calibration_audit(
            self.catalog,
            self.bank,
            RandomUtilityModel(
                beta=1.0,
                ranking_scale=0.35,
                default_scale=0.80,
                suggestion_scale=0.65,
            ),
            susceptibility_levels=(0.45,),
            split="test",
            planned_turns=4,
            domains=("travel",),
            policies=("balanced", "custom_adaptive"),
        )
        capacity = audit["capacity"]
        self.assertEqual(capacity["configured_domains"], ["travel"])
        self.assertEqual(
            capacity["non_cyclic_or_unknown_target_policies"],
            ["custom_adaptive"],
        )
        self.assertEqual(
            capacity["planned_capacity_basis"],
            "adaptive_or_unknown_policy_worst_case_all_turns_in_one_cell",
        )
        self.assertEqual(capacity["planned_per_cell_no_repeat_required"], 4)
        self.assertEqual(len(capacity["cells"]), 3)
        self.assertEqual(audit["catalog"]["selected_scenario_count"], 18)
        self.assertEqual(capacity["cyclic_reference_turns"], 16)
        self.assertEqual(
            capacity["cyclic_reference_per_cell_no_repeat_required"],
            6,
        )

    def test_exploratory_policy_has_balanced_coverage_capacity(self) -> None:
        audit = build_scenario_calibration_audit(
            self.catalog,
            self.bank,
            RandomUtilityModel(),
            susceptibility_levels=(0.45,),
            split="test",
            planned_turns=12,
            domains=("travel",),
            policies=("exploratory",),
        )
        capacity = audit["capacity"]
        self.assertEqual(
            capacity["non_cyclic_or_unknown_target_policies"],
            [],
        )
        self.assertEqual(
            capacity["planned_per_cell_no_repeat_required"],
            4,
        )
        self.assertTrue(capacity["planned_all_cells_sufficient"])

    def test_physical_order_probabilities_use_configured_minimum(self) -> None:
        audit = build_scenario_calibration_audit(
            self.catalog,
            self.bank,
            RandomUtilityModel(
                beta=1.0,
                ranking_scale=0.35,
                default_scale=0.80,
                suggestion_scale=0.65,
            ),
            susceptibility_levels=(0.45,),
            split="development",
            planned_turns=3,
            minimum_matched_probability=0.49,
        )
        guardrails = audit["probability_calibration"]["guardrails"]
        self.assertTrue(guardrails["order_averaged_balanced_probability"]["passed"])
        self.assertTrue(
            any(
                not item["passed"]
                and item["either_binary_response_at_or_below_minimum_count"] > 0
                for item in guardrails["physical_mechanism_probabilities"].values()
            )
        )
        self.assertFalse(guardrails["passed"])

    def test_cross_split_lexical_scan_covers_unselected_split_pair(self) -> None:
        source = next(
            scenario
            for scenario in self.catalog.scenarios
            if scenario.split == "development"
            and scenario.domain == "travel"
            and scenario.target_attribute == 0
        )
        target = next(
            scenario
            for scenario in self.catalog.scenarios
            if scenario.split == "train"
            and scenario.domain == "travel"
            and scenario.target_attribute == 0
        )
        replacement = self._near_copy_surface(
            target,
            source,
            marker="crosslexicalmarker",
            task_family=source.task_family,
        )
        synthetic = replace(
            self.catalog,
            scenarios=tuple(
                replacement if scenario.scenario_id == target.scenario_id else scenario
                for scenario in self.catalog.scenarios
            ),
        )
        audit = build_scenario_calibration_audit(
            synthetic,
            self.bank,
            RandomUtilityModel(),
            susceptibility_levels=(0.45,),
            split="test",
            planned_turns=3,
            domains=("travel",),
        )
        lexical = audit["warnings"]["lexical_overlap_candidates"]
        self.assertTrue(
            any(
                warning["kind"] == "cross_split_lexical_overlap_candidate"
                and {
                    warning["scenario_a"],
                    warning["scenario_b"],
                }
                == {source.scenario_id, target.scenario_id}
                and warning["semantic_similarity_claimed"] is False
                for warning in lexical
            )
        )
        family = audit["warnings"]["cross_split_exact_task_family_reuse"]
        self.assertTrue(
            any(
                warning["domain"] == "travel"
                and warning["task_family"] == source.task_family
                for warning in family
            )
        )

    def test_within_selected_split_lexical_warning_is_detected(self) -> None:
        candidates = [
            scenario
            for scenario in self.catalog.scenarios
            if scenario.split == "test"
            and scenario.domain == "travel"
            and scenario.target_attribute == 0
        ]
        source, target = candidates[:2]
        replacement = self._near_copy_surface(
            target,
            source,
            marker="withinlexicalmarker",
        )
        synthetic = replace(
            self.catalog,
            scenarios=tuple(
                replacement if scenario.scenario_id == target.scenario_id else scenario
                for scenario in self.catalog.scenarios
            ),
        )
        audit = build_scenario_calibration_audit(
            synthetic,
            self.bank,
            RandomUtilityModel(),
            susceptibility_levels=(0.45,),
            split="test",
            planned_turns=3,
            domains=("travel",),
        )
        self.assertTrue(
            any(
                warning["kind"] == "within_split_lexical_redundancy_candidate"
                and {
                    warning["scenario_a"],
                    warning["scenario_b"],
                }
                == {source.scenario_id, target.scenario_id}
                for warning in audit["warnings"]["lexical_overlap_candidates"]
            )
        )

    def test_within_unselected_split_pair_is_not_scanned(self) -> None:
        candidates = [
            scenario
            for scenario in self.catalog.scenarios
            if scenario.split == "development" and scenario.domain == "travel"
        ]
        source, target = candidates[:2]
        replacement = self._near_copy_surface(
            target,
            source,
            marker="unselectedwithinmarker",
        )
        synthetic = replace(
            self.catalog,
            scenarios=tuple(
                replacement if scenario.scenario_id == target.scenario_id else scenario
                for scenario in self.catalog.scenarios
            ),
        )
        audit = build_scenario_calibration_audit(
            synthetic,
            self.bank,
            RandomUtilityModel(),
            susceptibility_levels=(0.45,),
            split="test",
            planned_turns=3,
            domains=("travel",),
        )
        self.assertFalse(
            any(
                warning["kind"] == "within_split_lexical_redundancy_candidate"
                and {
                    warning["scenario_a"],
                    warning["scenario_b"],
                }
                == {source.scenario_id, target.scenario_id}
                for warning in audit["warnings"]["lexical_overlap_candidates"]
            )
        )

    def test_zero_presentation_effects_fail_only_effect_guardrails(self) -> None:
        audit = build_scenario_calibration_audit(
            self.catalog,
            self.bank,
            RandomUtilityModel(
                beta=1.0,
                ranking_scale=0.0,
                default_scale=0.0,
                suggestion_scale=0.0,
            ),
            susceptibility_levels=(0.45,),
            split="development",
            planned_turns=3,
        )
        guardrails = audit["probability_calibration"]["guardrails"]
        self.assertTrue(guardrails["order_averaged_balanced_probability"]["passed"])
        self.assertTrue(guardrails["order_averaged_restricted_probability"]["passed"])
        for mechanism in ("ranking", "default", "suggestion"):
            with self.subTest(mechanism=mechanism):
                effect = guardrails["mean_incremental_effects"][mechanism]
                self.assertEqual(effect["observed_mean"], 0.0)
                self.assertFalse(effect["passed"])
        self.assertTrue(audit["readiness"]["engineering_pilot"]["ready"])
        self.assertFalse(
            audit["readiness"]["scientific_pilot"]["criteria"][
                "prospective_probability_guardrails_pass"
            ]
        )

    def test_synthetic_long_label_creates_warning_without_mutation(self) -> None:
        reviewed_catalog = replace(
            self.catalog,
            scenarios=tuple(
                replace(
                    scenario,
                    review={
                        **scenario.review,
                        "automated_validation": "passed",
                        "surface_human_review": "passed",
                        "scientific_human_review": "passed",
                    },
                )
                if scenario.split == "development"
                else scenario
                for scenario in self.catalog.scenarios
            ),
        )
        target = next(
            scenario
            for scenario in reviewed_catalog.scenarios
            if scenario.split == "development"
            and scenario.domain == "travel"
            and scenario.target_attribute == 0
        )
        replacement = replace(
            target,
            positive_option=replace(
                target.positive_option,
                label=(
                    "A deliberately much longer premium upgraded rail seat "
                    "description with many extra unmatched words solely for "
                    "the synthetic symmetry-warning boundary review"
                ),
            ),
        )
        synthetic_catalog = replace(
            reviewed_catalog,
            scenarios=tuple(
                replacement if scenario.scenario_id == target.scenario_id else scenario
                for scenario in reviewed_catalog.scenarios
            ),
        )
        before = tuple(
            dict(scenario.review) for scenario in synthetic_catalog.scenarios
        )
        audit = build_scenario_calibration_audit(
            synthetic_catalog,
            self.bank,
            RandomUtilityModel(
                beta=1.0,
                ranking_scale=0.35,
                default_scale=0.80,
                suggestion_scale=0.65,
            ),
            susceptibility_levels=(0.45,),
            split="development",
            planned_turns=3,
        )
        after = tuple(dict(scenario.review) for scenario in synthetic_catalog.scenarios)
        self.assertEqual(after, before)
        warnings = audit["warnings"]["raw_option_label_word_count_warnings"]
        self.assertTrue(
            any(
                warning["scenario_id"] == target.scenario_id
                and warning["kind"] == "option_label_raw_word_count_difference"
                and warning["absolute_difference"] > 2
                for warning in warnings
            )
        )
        self.assertTrue(
            any(
                warning["scenario_id"] == target.scenario_id
                and warning["kind"] == "option_label_raw_word_count_ratio_outside_range"
                for warning in warnings
            )
        )
        self.assertFalse(audit["warnings"]["blocks_machine_readiness"])
        self.assertTrue(audit["warnings"]["blocks_recorded_scientific_readiness"])
        self.assertFalse(
            audit["readiness"]["scientific_pilot"]["criteria"][
                "unresolved_machine_warning_count_is_zero"
            ]
        )
        self.assertEqual(
            audit["readiness"]["scientific_pilot"]["blocking_reasons"],
            [
                "independent_human_review_evidence_bundle_verified",
                "unresolved_machine_warning_count_is_zero",
            ],
        )

    def test_rejects_invalid_planning_inputs(self) -> None:
        model = RandomUtilityModel()
        cases = (
            (
                {"susceptibility_levels": (), "split": "test", "planned_turns": 3},
                ValueError,
                "cannot be empty",
            ),
            (
                {
                    "susceptibility_levels": (0.2, 0.2),
                    "split": "test",
                    "planned_turns": 3,
                },
                ValueError,
                "must be distinct",
            ),
            (
                {
                    "susceptibility_levels": (False,),
                    "split": "test",
                    "planned_turns": 3,
                },
                TypeError,
                "must be numeric",
            ),
            (
                {
                    "susceptibility_levels": (0.2,),
                    "split": "unknown",
                    "planned_turns": 3,
                },
                ValueError,
                "split must",
            ),
            (
                {
                    "susceptibility_levels": (0.2,),
                    "split": "test",
                    "planned_turns": True,
                },
                ValueError,
                "positive integer",
            ),
            (
                {
                    "susceptibility_levels": (0.2,),
                    "split": "test",
                    "planned_turns": 3,
                    "domains": (),
                },
                ValueError,
                "domains cannot be empty",
            ),
            (
                {
                    "susceptibility_levels": (0.2,),
                    "split": "test",
                    "planned_turns": 3,
                    "domains": ("unknown",),
                },
                ValueError,
                "unknown domain",
            ),
            (
                {
                    "susceptibility_levels": (0.2,),
                    "split": "test",
                    "planned_turns": 3,
                    "policies": ("balanced", "balanced"),
                },
                ValueError,
                "policies must be distinct",
            ),
            (
                {
                    "susceptibility_levels": (0.2,),
                    "split": "test",
                    "planned_turns": 3,
                    "minimum_matched_probability": 0.5,
                },
                ValueError,
                "strictly between 0 and 0.5",
            ),
        )
        for kwargs, exception, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(exception, message):
                    build_scenario_calibration_audit(
                        self.catalog,
                        self.bank,
                        model,
                        **kwargs,
                    )

    def test_markdown_rejects_unrecognized_audit_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema_version"):
            render_scenario_calibration_markdown(
                {
                    "schema_version": 999,
                    "audit_policy": AUDIT_POLICY,
                }
            )
        with self.assertRaisesRegex(ValueError, "schema_version"):
            render_blinded_surface_review_markdown(
                {
                    "schema_version": 999,
                    "audit_policy": AUDIT_POLICY,
                }
            )
        with self.assertRaisesRegex(ValueError, "audit_policy"):
            render_scenario_calibration_markdown(
                {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "audit_policy": "unknown",
                }
            )
        with self.assertRaisesRegex(ValueError, "audit_policy"):
            render_blinded_surface_review_markdown(
                {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "audit_policy": "unknown",
                }
            )


if __name__ == "__main__":
    unittest.main()
