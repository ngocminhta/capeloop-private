from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from cape_loop.artifacts import file_sha256
from cape_loop.config import load_config
from cape_loop.cli import main
from cape_loop.conversation_surfaces import load_conversation_bank
from cape_loop.response import RandomUtilityModel
from cape_loop.scenario_calibration import build_scenario_calibration_audit
from cape_loop.scenario_review import (
    build_scenario_review_kit,
    derive_reviewed_catalog,
    verify_scenario_review_evidence,
)
from cape_loop.scenarios import load_scenario_catalog


ROOT = Path(__file__).resolve().parents[1]


class ScenarioReviewEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(ROOT / "configs/live/experiment_b_openrouter.toml")
        cls.loaded = load_scenario_catalog(
            config.scenarios.catalog_file,
            expected_sha256=config.scenarios.catalog_sha256,
        )
        cls.bank = load_conversation_bank(config.scenarios.conversation_file)
        response = config.response_model
        model = RandomUtilityModel(
            beta=response.beta / response.decision_noise,
            ranking_scale=response.rank_scale / response.decision_noise,
            default_scale=response.default_scale / response.decision_noise,
            suggestion_scale=response.suggestion_scale / response.decision_noise,
        )
        cls.audit = build_scenario_calibration_audit(
            cls.loaded.catalog,
            cls.bank,
            model,
            susceptibility_levels=response.susceptibility_levels,
            split="all",
            planned_turns=6,
            domains=config.experiment.domains,
            policies=config.experiment.policies,
            minimum_matched_probability=response.minimum_matched_probability,
        )
        cls.kit = build_scenario_review_kit(
            cls.loaded.catalog,
            cls.bank,
            cls.audit,
            catalog_sha256=cls.loaded.source_sha256,
            conversation_bank_sha256=file_sha256(
                Path(config.scenarios.conversation_file)
            ),
        )

    def _passing_evidence(self) -> list[dict[str, object]]:
        evidence: list[dict[str, object]] = []
        for reviewer_id in ("surface-reviewer-1", "surface-reviewer-2"):
            review = deepcopy(self.kit["surface_template"])
            review["reviewer"] = {
                "reviewer_id": reviewer_id,
                "independent_from_authors_and_other_reviewers": True,
                "outcome_blind": True,
            }
            for item in review["items"]:
                for rating in item["ratings"]:
                    rating["naturalness"] = 5
                    rating["neutrality"] = 5
                item["assertions"] = {
                    "grammatical_and_coherent": True,
                    "comparable_clarity_and_specificity": True,
                    "no_unexplained_choice_pressure": True,
                }
            evidence.append(review)

        scientific_ids = ("scientific-reviewer-1", "scientific-reviewer-2")
        positive_option_by_item = {
            item["item_id"]: ("a" if item["option_a_role"] == "positive" else "b")
            for item in self.kit["mapping"]["items"]
        }
        for reviewer_id in scientific_ids:
            review = deepcopy(self.kit["scientific_template"])
            review["reviewer"] = {
                "reviewer_id": reviewer_id,
                "independent_from_authors_and_other_reviewers": True,
                "outcome_blind": True,
            }
            for item in review["items"]:
                for mapping in item["fact_mapping"]:
                    mapping.update(
                        {
                            "visible_fact_count": 1,
                            "mapped_fact_count": 1,
                            "unmodeled_fact_count": 0,
                            "ambiguous_fact_count": 0,
                            "cross_loading_fact_count": 0,
                            "assigned_roles": ["target"],
                        }
                    )
                item["judgments"] = {
                    "recovered_positive_option": positive_option_by_item[
                        item["item_id"]
                    ],
                    "unintended_dimension_assignment_count": 0,
                    "tradeoff_valid": True,
                    "non_dominating": True,
                    "treatment_isolated": True,
                }
                item["masking_review"] = {
                    "masked_option_a": "Masked description A",
                    "masked_option_b": "Masked description B",
                    "target_language_removed": True,
                    "non_target_facts_preserved": True,
                }
                for warning in item["warning_dispositions"]:
                    warning["disposition"] = "resolved_valid"
                    warning["rationale"] = "Independent semantic inspection passed."
            evidence.append(review)

        choice = deepcopy(self.kit["neutral_choice_template"])
        choice["collection"] = {
            "collector_id": "choice-collector",
            "preregistration_id": "choice-pretest-v1",
            "collected_before_evaluated_model_outcomes": True,
            "exclusions_applied_as_preregistered": True,
            "independent_participants": True,
        }
        for item in choice["items"]:
            item["responses"] = [
                {
                    "participant_id": f"choice-participant-{index:03d}",
                    "display_order": "a_first" if index % 2 == 0 else "b_first",
                    "selected_option": "a" if index % 2 == 0 else "b",
                }
                for index in range(40)
            ]
        evidence.append(choice)

        attractiveness = deepcopy(self.kit["attractiveness_template"])
        attractiveness["collection"] = {
            "collector_id": "attractiveness-collector",
            "preregistration_id": "attractiveness-pretest-v1",
            "collected_before_evaluated_model_outcomes": True,
            "exclusions_applied_as_preregistered": True,
            "independent_participants": True,
            "masking_scientific_reviewer_ids": list(scientific_ids),
        }
        for item in attractiveness["items"]:
            item["masked_option_a"] = "Masked description A"
            item["masked_option_b"] = "Masked description B"
            item["target_language_masked"] = True
            item["responses"] = [
                {
                    "participant_id": f"attractiveness-participant-{index:03d}",
                    "display_order": "a_first" if index % 2 == 0 else "b_first",
                    "rating_a": 3 + index % 2,
                    "rating_b": 3 + index % 2,
                }
                for index in range(80)
            ]
        evidence.append(attractiveness)
        return evidence

    def test_complete_independent_evidence_promotes_a_new_valid_catalog(self) -> None:
        evidence = self._passing_evidence()
        report = verify_scenario_review_evidence(self.kit, evidence, self.audit)
        self.assertTrue(report["promotion_eligible"])
        self.assertFalse(report["catalog_review_strings_used_as_evidence"])
        source = json.loads(self.loaded.source_bytes.decode("utf-8"))
        reviewed = derive_reviewed_catalog(
            source,
            report,
            new_catalog_version="1.6.0-reviewed-test",
            frozen_on="2026-08-01",
        )
        self.assertEqual(reviewed["catalog_status"], "frozen-paper")
        self.assertTrue(
            all(row["status"] == "approved" for row in reviewed["scenarios"])
        )

    def test_reviewer_overlap_is_rejected(self) -> None:
        evidence = self._passing_evidence()
        evidence[2]["reviewer"]["reviewer_id"] = "surface-reviewer-1"
        with self.assertRaisesRegex(ValueError, "reviewer sets must be distinct"):
            verify_scenario_review_evidence(self.kit, evidence, self.audit)

    def test_cli_writes_a_fresh_derived_release_without_source_mutation(self) -> None:
        source_before = self.loaded.source_bytes
        with TemporaryDirectory() as directory:
            root = Path(directory)
            audit_dir = root / "audit"
            evidence_dir = root / "evidence"
            output_dir = root / "reviewed"
            audit_dir.mkdir()
            evidence_dir.mkdir()
            for filename, payload in (
                ("scenario-audit.json", self.audit),
                ("review-protocol.json", self.kit["protocol"]),
                ("review-item-map.json", self.kit["mapping"]),
            ):
                (audit_dir / filename).write_text(json.dumps(payload), encoding="utf-8")
            counts: dict[str, int] = {}
            for payload in self._passing_evidence():
                kind = str(payload["evidence_kind"])
                counts[kind] = counts.get(kind, 0) + 1
                (evidence_dir / f"{kind}-{counts[kind]}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "scenarios",
                        "review-promote",
                        str(ROOT / "configs" / "live" / "experiment_b_openrouter.toml"),
                        str(audit_dir),
                        str(evidence_dir),
                        str(output_dir),
                        "--catalog-version",
                        "1.6.0-reviewed-cli-test",
                        "--frozen-on",
                        "2026-08-01",
                    ]
                )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(summary["promotion_eligible"])
            self.assertTrue((output_dir / "reviewed-scenario-catalog.json").is_file())
            reviewed_bank_path = output_dir / "reviewed-conversation-templates.json"
            reviewed_bank = load_conversation_bank(reviewed_bank_path)
            self.assertNotEqual(reviewed_bank.bank_id, self.bank.bank_id)
            self.assertIn("independently-reviewed-under", reviewed_bank.source)
            self.assertTrue(
                all(
                    "independently-reviewed-under" in template.source
                    for template in reviewed_bank.templates
                )
            )
            self.assertEqual(self.loaded.source_path.read_bytes(), source_before)

            tampered = deepcopy(self.audit)
            tampered["probability_calibration"]["all_cells_passed"] = False
            (audit_dir / "scenario-audit.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "scenarios",
                        "review-promote",
                        str(ROOT / "configs" / "live" / "experiment_b_openrouter.toml"),
                        str(audit_dir),
                        str(evidence_dir),
                        str(root / "tampered-output"),
                        "--catalog-version",
                        "1.6.0-tampered-test",
                        "--frozen-on",
                        "2026-08-01",
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("does not equal the audit recomputed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
