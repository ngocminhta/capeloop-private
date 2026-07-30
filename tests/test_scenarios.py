from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from cape_loop.config import load_config
from cape_loop.domains import DATA_SPLITS, get_domain
from cape_loop.elicitation import build_matched_anchor_set
from cape_loop.experiments import generate_fixed_history, run_trajectory
from cape_loop.policies import BalancedPolicy
from cape_loop.scenarios import (
    ScenarioCatalog,
    load_scenario_catalog,
    materialize_context,
)
from cape_loop.schemas import LatentUser, Susceptibility
from cape_loop.updaters import NoUpdateUpdater

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPOSITORY_ROOT / "data" / "scenarios" / "scenario-catalog-v1.json"
CATALOG_SHA256 = "7b7144b3b3f75ac7284ab6153d1b6ce62cf293aec94004ee2cb3111bcc1f6cf1"


def _canonical_payload() -> dict[str, object]:
    decoded = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise AssertionError("canonical scenario catalog must be an object")
    return decoded


class CanonicalScenarioCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_scenario_catalog(
            CATALOG_PATH,
            expected_sha256=CATALOG_SHA256,
        )
        cls.catalog = cls.loaded.catalog

    def test_canonical_digest_load_and_complete_48_row_coverage(self) -> None:
        material = CATALOG_PATH.read_bytes()
        self.assertEqual(sha256(material).hexdigest(), CATALOG_SHA256)
        self.assertEqual(self.loaded.source_bytes, material)
        self.assertEqual(self.loaded.source_sha256, CATALOG_SHA256)

        report = self.catalog.coverage_report()
        self.assertEqual(report["scenario_count"], 48)
        self.assertEqual(report["family_count"], 48)
        self.assertEqual(len(report["cells"]), 18)

        observed_cells = {
            (
                cell["domain"],
                cell["split"],
                cell["target_attribute"],
            ): cell["scenario_count"]
            for cell in report["cells"]
        }
        expected_cells = {
            (domain, split, target): (6 if split == "test" else 1)
            for domain in ("travel", "writing")
            for split in DATA_SPLITS
            for target in range(3)
        }
        self.assertEqual(observed_cells, expected_cells)

    def test_selection_is_deterministic_and_catalog_order_independent(self) -> None:
        reversed_catalog = replace(
            self.catalog,
            scenarios=tuple(reversed(self.catalog.scenarios)),
        )
        selections = []
        for key in (
            ("trajectory", 0),
            ("trajectory", 1),
            {"trajectory": "paired", "turn": 2},
        ):
            with self.subTest(selection_key=key):
                first = self.catalog.select(
                    domain="travel",
                    split="test",
                    target_attribute=0,
                    seed=1729,
                    selection_key=key,
                )
                repeated = self.catalog.select(
                    domain="travel",
                    split="test",
                    target_attribute=0,
                    seed=1729,
                    selection_key=key,
                )
                reordered = reversed_catalog.select(
                    domain="travel",
                    split="test",
                    target_attribute=0,
                    seed=1729,
                    selection_key=key,
                )
                self.assertEqual(first.scenario_id, repeated.scenario_id)
                self.assertEqual(first.scenario_id, reordered.scenario_id)
                selections.append(first.scenario_id)
        self.assertTrue(
            set(selections)
            <= {
                scenario.scenario_id
                for scenario in self.catalog.eligible("travel", "test", 0)
            }
        )

    def test_selection_respects_every_split_and_attribute_stratum(self) -> None:
        for domain in ("travel", "writing"):
            for split in DATA_SPLITS:
                for target in range(3):
                    with self.subTest(
                        domain=domain,
                        split=split,
                        target=target,
                    ):
                        scenario = self.catalog.select(
                            domain=domain,
                            split=split,
                            target_attribute=target,
                            seed=31,
                            selection_key=("coverage", domain, split, target),
                        )
                        self.assertEqual(scenario.domain, domain)
                        self.assertEqual(scenario.split, split)
                        self.assertEqual(scenario.target_attribute, target)
                        self.assertEqual(
                            scenario.target_key,
                            get_domain(domain).attributes[target].key,
                        )

    def test_cycle_selection_uses_each_scenario_before_repeating(self) -> None:
        available = self.catalog.eligible("travel", "test", 0)
        selected = tuple(
            self.catalog.select_cycle(
                domain="travel",
                split="test",
                target_attribute=0,
                seed=1729,
                cycle_key=("trajectory", "same-counterfactual"),
                occurrence_index=index,
            )
            for index in range(7)
        )
        self.assertEqual(
            {item.scenario_id for item in selected[:6]},
            {scenario.scenario_id for scenario in available},
        )
        self.assertEqual(selected[0].scenario_id, selected[6].scenario_id)
        repeated = self.catalog.select_cycle(
            domain="travel",
            split="test",
            target_attribute=0,
            seed=1729,
            cycle_key=("trajectory", "same-counterfactual"),
            occurrence_index=5,
        )
        self.assertEqual(selected[5], repeated)

        with self.assertRaisesRegex(ValueError, "occurrence_index"):
            self.catalog.select_cycle(
                domain="travel",
                split="test",
                target_attribute=0,
                seed=1729,
                cycle_key="trajectory",
                occurrence_index=-1,
            )

    def test_materialization_preserves_treatments_and_uses_catalog_surfaces(
        self,
    ) -> None:
        scenario = self.catalog.select(
            domain="travel",
            split="test",
            target_attribute=0,
            seed=1729,
            selection_key="materialization",
        )
        matched = build_matched_anchor_set(
            get_domain("travel"),
            target_attribute=0,
            anchor_direction=-1,
            scenario_id="generic",
        )

        expected_by_mechanism = {
            "balanced": (
                scenario.negative_option,
                scenario.positive_option,
            ),
            "restricted": (
                scenario.negative_option,
                scenario.negative_same_direction_option,
            ),
            "default": (
                scenario.negative_option,
                scenario.positive_option,
            ),
            "suggested": (
                scenario.negative_option,
                scenario.positive_option,
            ),
        }
        for mechanism, expected_options in expected_by_mechanism.items():
            with self.subTest(mechanism=mechanism):
                generic = matched.context(mechanism)
                context = materialize_context(generic, scenario)
                self.assertEqual(context.scenario_id, scenario.scenario_id)
                self.assertEqual(
                    context.wording_template,
                    scenario.wording_template_id,
                )
                self.assertEqual(context.prompt, scenario.prompt)
                self.assertEqual(
                    context.option_ids,
                    tuple(option.option_id for option in expected_options),
                )
                self.assertEqual(
                    tuple(option.label for option in context.options),
                    tuple(option.label for option in expected_options),
                )
                self.assertEqual(
                    context.ranking,
                    tuple(option.option_id for option in expected_options),
                )
                self.assertEqual(
                    context.default_option_id,
                    (
                        scenario.negative_option.option_id
                        if mechanism == "default"
                        else None
                    ),
                )
                self.assertEqual(
                    context.suggested_option_id,
                    (
                        scenario.negative_option.option_id
                        if mechanism == "suggested"
                        else None
                    ),
                )

        self.assertEqual(
            matched.context("default").default_option_id,
            matched.anchor_option_id,
        )
        self.assertNotEqual(
            matched.context("default").default_option_id,
            scenario.negative_option.option_id,
        )

    def test_catalog_honestly_reports_not_paper_eligible(self) -> None:
        report = self.catalog.coverage_report()
        self.assertEqual(report["catalog_status"], "frozen-development")
        self.assertEqual(report["eligibility"], "simulation-and-pilot-only")
        self.assertEqual(report["approved_scenario_count"], 0)
        self.assertEqual(report["provisional_scenario_count"], 48)
        self.assertFalse(report["paper_eligible"])
        self.assertFalse(self.loaded.input_manifest()["paper_eligible"])
        self.assertTrue(
            all(
                scenario.status == "provisional"
                and scenario.review["paper_eligible"] is False
                and scenario.review["surface_human_review"] == "not_completed"
                and scenario.review["scientific_human_review"] == "not_completed"
                for scenario in self.catalog.scenarios
            )
        )

    def test_closed_loop_and_fixed_history_use_requested_catalog_split(
        self,
    ) -> None:
        user = LatentUser(
            "catalog-user",
            (2, -1, 1),
            Susceptibility(ranking=0.35, default=0.80, suggestion=0.65),
        )
        trajectory = run_trajectory(
            user=user,
            domain=get_domain("travel"),
            policy=BalancedPolicy(),
            updater=NoUpdateUpdater(),
            turns=3,
            seed=31,
            scenario_catalog=self.catalog,
            data_split="development",
            crn_key="catalog-closed-loop",
        )
        development_contexts = tuple(
            interaction.context for interaction in trajectory.audit_record.interactions
        )
        self.assertEqual(
            tuple(context.target_attribute for context in development_contexts),
            (0, 1, 2),
        )
        self.assertTrue(
            all(
                self.catalog.scenario(context.scenario_id).split == "development"
                for context in development_contexts
            )
        )

        history = generate_fixed_history(
            user=user,
            domain=get_domain("travel"),
            policy=BalancedPolicy(),
            turns=6,
            seed=31,
            scenario_catalog=self.catalog,
            data_split="test",
            crn_key="catalog-fixed-history",
        )
        self.assertEqual(
            tuple(event.context.target_attribute for event in history.events),
            (0, 1, 2, 0, 1, 2),
        )
        self.assertTrue(
            all(
                self.catalog.scenario(event.context.scenario_id).split == "test"
                for event in history.events
            )
        )
        for target in range(3):
            target_scenarios = {
                event.context.scenario_id
                for event in history.events
                if event.context.target_attribute == target
            }
            self.assertEqual(len(target_scenarios), 2)

        long_trajectory = run_trajectory(
            user=user,
            domain=get_domain("travel"),
            policy=BalancedPolicy(),
            updater=NoUpdateUpdater(),
            turns=16,
            seed=31,
            scenario_catalog=self.catalog,
            data_split="test",
            crn_key="catalog-long-closed-loop",
        )
        actual_scenario_ids = tuple(
            interaction.context.scenario_id
            for interaction in long_trajectory.audit_record.interactions
        )
        self.assertEqual(len(actual_scenario_ids), 16)
        self.assertEqual(len(set(actual_scenario_ids)), 16)
        self.assertTrue(
            all(
                not turn.profile_influenced_action
                for turn in long_trajectory.turns
            )
        )

    def test_checked_in_configs_bind_catalog_and_cover_fitted_cells(
        self,
    ) -> None:
        paths = sorted(
            path
            for path in (REPOSITORY_ROOT / "configs").glob("**/*.toml")
            if "local" not in path.relative_to(
                REPOSITORY_ROOT / "configs"
            ).parts
        )
        paths.append(
            REPOSITORY_ROOT
            / "analysis"
            / "confirmatory-mixed-effects"
            / "fixtures"
            / "confirmatory_ci.toml"
        )
        self.assertEqual(len(paths), 19)
        for path in paths:
            with self.subTest(config=path.relative_to(REPOSITORY_ROOT)):
                config = load_config(path)
                self.assertEqual(
                    config.scenarios.catalog_file,
                    "data/scenarios/scenario-catalog-v1.json",
                )
                self.assertEqual(
                    config.scenarios.catalog_sha256,
                    CATALOG_SHA256,
                )
                self.assertGreaterEqual(
                    config.inference.training_interactions
                    // len(config.experiment.domains),
                    24,
                )


class ScenarioCatalogValidationTests(unittest.TestCase):
    def test_checksum_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "scenario catalog SHA-256 mismatch",
        ):
            load_scenario_catalog(
                CATALOG_PATH,
                expected_sha256="0" * 64,
            )

    def test_unknown_fields_are_rejected_at_each_record_level(self) -> None:
        mutations = {
            "catalog": lambda payload: payload.__setitem__("unknown", True),
            "scenario": lambda payload: payload["scenarios"][0].__setitem__(
                "unknown",
                True,
            ),
            "option": lambda payload: payload["scenarios"][0][
                "negative_option"
            ].__setitem__("unknown", True),
        }
        for level, mutate in mutations.items():
            with self.subTest(level=level):
                payload = deepcopy(_canonical_payload())
                mutate(payload)
                with self.assertRaisesRegex(
                    ValueError,
                    "fields must be exact",
                ):
                    ScenarioCatalog.parse(payload)

    def test_duplicate_json_object_keys_are_rejected(self) -> None:
        material = b'{"schema_version":1,"schema_version":1}\n'
        with TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-keys.json"
            path.write_bytes(material)
            with self.assertRaisesRegex(ValueError, "contains duplicate key"):
                load_scenario_catalog(
                    path,
                    expected_sha256=sha256(material).hexdigest(),
                )

    def test_malformed_feature_contract_is_rejected(self) -> None:
        mutations = {
            "wrong_cardinality": [-0.5, 0.0],
            "wrong_semantic_vector": [-0.25, 0.0, 0.0],
            "boolean_component": [False, 0.0, 0.0],
        }
        for label, features in mutations.items():
            with self.subTest(case=label):
                payload = deepcopy(_canonical_payload())
                payload["scenarios"][0]["negative_option"]["features"] = features
                with self.assertRaisesRegex(ValueError, "features must"):
                    ScenarioCatalog.parse(payload)

    def test_scenario_families_cannot_cross_splits(self) -> None:
        payload = deepcopy(_canonical_payload())
        train = next(
            scenario
            for scenario in payload["scenarios"]
            if scenario["split"] == "train"
        )
        development = next(
            scenario
            for scenario in payload["scenarios"]
            if scenario["split"] == "development"
        )
        development["family_id"] = train["family_id"]
        with self.assertRaisesRegex(
            ValueError,
            "scenario families cross data splits",
        ):
            ScenarioCatalog.parse(payload)

    def test_approved_and_paper_eligible_rows_require_completed_reviews(
        self,
    ) -> None:
        payload = deepcopy(_canonical_payload())
        scenario = payload["scenarios"][0]
        scenario["status"] = "approved"
        scenario["review"]["paper_eligible"] = True
        with self.assertRaisesRegex(
            ValueError,
            "approved scenarios require every review to pass",
        ):
            ScenarioCatalog.parse(payload)

    def test_reviewed_catalog_has_an_attainable_paper_freeze_state(self) -> None:
        payload = deepcopy(_canonical_payload())
        payload["catalog_status"] = "frozen-paper"
        payload["eligibility"] = "paper-eligible"
        for scenario in payload["scenarios"]:
            scenario["status"] = "approved"
            scenario["review"]["automated_validation"] = "passed"
            scenario["review"]["surface_human_review"] = "passed"
            scenario["review"]["scientific_human_review"] = "passed"
            scenario["review"]["paper_eligible"] = True
        parsed = ScenarioCatalog.parse(payload)
        self.assertEqual(parsed.catalog_status, "frozen-paper")
        self.assertEqual(parsed.eligibility, "paper-eligible")
        self.assertTrue(
            all(
                scenario.status == "approved"
                and scenario.review["paper_eligible"]
                for scenario in parsed.scenarios
            )
        )

    def test_paper_freeze_rejects_incomplete_or_inconsistent_catalog(self) -> None:
        payload = deepcopy(_canonical_payload())
        payload["catalog_status"] = "frozen-paper"
        with self.assertRaisesRegex(
            ValueError,
            "frozen-paper catalogs must declare paper-eligible",
        ):
            ScenarioCatalog.parse(payload)

        payload = deepcopy(_canonical_payload())
        payload["eligibility"] = "paper-eligible"
        with self.assertRaisesRegex(
            ValueError,
            "frozen-development catalogs must declare",
        ):
            ScenarioCatalog.parse(payload)

        payload = deepcopy(_canonical_payload())
        payload["catalog_status"] = "frozen-paper"
        payload["eligibility"] = "paper-eligible"
        with self.assertRaisesRegex(
            ValueError,
            "frozen-paper catalogs require every scenario",
        ):
            ScenarioCatalog.parse(payload)

    def test_runtime_constants_match_the_published_catalog_schema(self) -> None:
        for field, value, message in (
            ("source", "unknown-source", "source must"),
            ("split_policy", "unknown-policy", "split_policy must"),
        ):
            with self.subTest(field=field):
                payload = deepcopy(_canonical_payload())
                payload[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    ScenarioCatalog.parse(payload)

    def test_wording_templates_cannot_cross_splits(self) -> None:
        payload = deepcopy(_canonical_payload())
        train = next(
            scenario
            for scenario in payload["scenarios"]
            if scenario["split"] == "train"
        )
        development = next(
            scenario
            for scenario in payload["scenarios"]
            if scenario["split"] == "development"
        )
        development["wording_template_id"] = train["wording_template_id"]
        with self.assertRaisesRegex(
            ValueError,
            "wording templates cross data splits",
        ):
            ScenarioCatalog.parse(payload)


if __name__ == "__main__":
    unittest.main()
