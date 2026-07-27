from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
import json
import unittest

from cape_loop.artifacts import RunArtifacts
from cape_loop.cli import main as cli_main
from cape_loop.config import load_config
from cape_loop.h7_control_review import (
    BASELINE_UPDATER_ID,
    BINDINGS_FILENAME,
    CLAIM_STATUS,
    MITIGATION_UPDATER_ID,
    PLAN_FILENAME,
    REQUESTS_FILENAME,
    build_h7_volunteered_collection_plan,
    create_h7_volunteered_review,
    load_verified_h7_source,
    snapshot_h7_review_inputs,
    verify_h7_volunteered_review,
    volunteered_updates_from_provider_evidence,
    write_h7_plan_directory,
)
from cape_loop.llm_exchange import ATTRIBUTES, VALUES, LLMResponse
from cape_loop.schema_export import SCHEMAS


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _population_rows() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "schema_version": 1,
            "user_id": f"test-user-{index}",
            "domain": "travel",
            "theta": theta,
            "susceptibility": {
                "ranking": 0.15,
                "default": 0.15,
                "suggestion": 0.15,
            },
            "split": "test",
        }
        for index, theta in enumerate(
            ((-2, 1, 2), (2, -1, -2)),
            start=1,
        )
    )


def _source_binding() -> dict[str, str]:
    return {
        "run_id": "source-run",
        "manifest_sha256": "a" * 64,
        "config_file_sha256": "b" * 64,
        "checksums_sha256": "c" * 64,
        "population_sha256": "d" * 64,
        "experiment_a_metrics_sha256": "e" * 64,
        "hypothesis_estimands_sha256": "f" * 64,
    }


def _beliefs_for(binding, mass: float = 0.80) -> dict[str, dict[str, float]]:
    case_direction = (
        -1
        if "test-user-1" in binding.case_id
        and binding.case_id.endswith("attribute-1")
        else None
    )
    # The exact direction comes from the model-visible statement only in this
    # fixture helper; production conversion uses the withheld case binding.
    statement = binding.llm_request.payload["observation"][
        "event_sequence"
    ][0]["surface_response"]
    negative_labels = {"budget", "central", "convenience"}
    direction = (
        -1
        if any(label in statement for label in negative_labels)
        else 1
    )
    del case_direction
    target = binding.llm_request.payload["observation"]["target_attribute"]
    beliefs: dict[str, dict[str, float]] = {}
    for index, attribute in enumerate(ATTRIBUTES):
        if index != target:
            beliefs[attribute] = {value: 0.25 for value in VALUES}
            continue
        same = mass / 2.0
        opposite = (1.0 - mass) / 2.0
        beliefs[attribute] = {
            "-2": same if direction < 0 else opposite,
            "-1": same if direction < 0 else opposite,
            "+1": same if direction > 0 else opposite,
            "+2": same if direction > 0 else opposite,
        }
    return beliefs


def _provider_material(plan):
    responses = []
    audits = []
    for binding in plan.request_bindings:
        raw_digest = sha256(
            binding.llm_request.request_id.encode("utf-8")
        ).hexdigest()
        response = LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": binding.llm_request.request_id,
                "prompt_sha256": binding.llm_request.prompt_sha256,
                "model_id": "provider-model",
                "beliefs": _beliefs_for(binding),
                "raw_response_sha256": raw_digest,
            }
        )
        responses.append(response)
        audits.append(
            {
                "schema_version": 1,
                "provider": "openai",
                "acceptance_status": "accepted",
                "request_id": response.request_id,
                "prompt_sha256": response.prompt_sha256,
                "request_body_sha256": sha256(
                    ("body:" + response.request_id).encode("utf-8")
                ).hexdigest(),
                "model_requested": "provider-model",
                "model_returned": "provider-model",
                "provider_response_id": "response:" + response.request_id,
                "attempts": 1,
                "raw_response_sha256": raw_digest,
                "raw_response": {"id": "response:" + response.request_id},
                "replay_response": response.to_dict(),
            }
        )
    return tuple(responses), tuple(audits)


def _source_h7_artifact() -> dict[str, Any]:
    superiority = [
        {
            "mechanism": mechanism,
            "criterion_met": True,
            "computed_status": "criterion_met",
        }
        for mechanism in ("restricted", "default", "suggested")
    ]
    valid_learning = {
        "condition": "balanced",
        "pair_count": 4,
        "baseline_directional_update": {"estimate": 1.0},
        "mitigation_directional_update": {"estimate": 0.9},
        "retention_contrast": {"estimate": 0.1},
        "retention_fraction": 0.8,
        "minimum_cluster_count": 2,
        "criterion": "retained valid learning",
        "criterion_met": True,
        "computed_status": "criterion_met",
        "missing_reason": None,
    }
    missing_volunteered = {
        "condition": "volunteered",
        "pair_count": 0,
        "baseline_directional_update": None,
        "mitigation_directional_update": None,
        "retention_contrast": None,
        "retention_fraction": 0.8,
        "minimum_cluster_count": 2,
        "criterion": "retained valid learning",
        "criterion_met": None,
        "computed_status": "incomplete",
        "missing_reason": "not supplied",
    }
    return {
        "schema_version": 1,
        "analysis": "experiment_a_hypothesis_estimands",
        "independent_unit": "complete latent user",
        "bootstrap_replicates": 40,
        "confidence_level": 0.95,
        "frozen_decision_constants": {
            "policy_conditioned_mechanisms": [
                "restricted",
                "default",
                "suggested",
            ],
            "minimum_cluster_count": 2,
            "h2_required_mechanisms": 2,
            "h7_required_superiority_mechanisms": 2,
            "h7_valid_learning_retention_fraction": 0.8,
        },
        "hypotheses": {
            "H1": {},
            "H2": {},
            "H7": {
                "hypothesis_id": "H7",
                "name": "Causal provenance is actionable",
                "component": "experiment_a_update_error_and_valid_learning",
                "baseline_updater_id": BASELINE_UPDATER_ID,
                "mitigation_updater_id": MITIGATION_UPDATER_ID,
                "response_mode": "controlled_anchor",
                "required_superiority_mechanisms": 2,
                "qualifying_superiority_mechanisms": [
                    "restricted",
                    "default",
                    "suggested",
                ],
                "missing_superiority_mechanisms": [],
                "inadequate_cluster_mechanisms": [],
                "superiority_estimands": superiority,
                "balanced_valid_learning": valid_learning,
                "volunteered_valid_learning": missing_volunteered,
                "retention_noninferiority_margin": {
                    "retention_fraction": 0.8,
                    "maximum_relative_loss": 0.2,
                },
                "complete": False,
                "criterion": "all H7 Experiment A components",
                "criterion_met": None,
                "computed_status": "incomplete",
                "claim_status": CLAIM_STATUS,
                "scope_note": "Experiment B remains separate.",
            },
        },
        "claim_status": CLAIM_STATUS,
        "interpretation": "Non-claiming fixture.",
    }


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _verified_source(root: Path) -> Path:
    config = load_config(_REPO_ROOT / "configs" / "openai_primary.toml")
    run = RunArtifacts.create(config, root=root)
    run.write_jsonl("population/users.jsonl", _population_rows())
    metric_rows = [
        {
            "trial_id": f"{user_id}:{domain}:controlled",
            "user_id": user_id,
            "domain": domain,
            "response_mode": "controlled_anchor",
            "updater_id": updater_id,
        }
        for user_id, domain in (
            (str(row["user_id"]), str(row["domain"]))
            for row in _population_rows()
        )
        for updater_id in (BASELINE_UPDATER_ID, MITIGATION_UPDATER_ID)
    ]
    run.write_jsonl("metrics/experiment-a.jsonl", metric_rows)
    run.write_json(
        "metrics/experiment-a-hypothesis-estimands.json",
        _source_h7_artifact(),
    )
    run.finalize(
        {
            "experiment": "A",
            "scientific_claim_status": "not_claimed",
        }
    )
    return run.path


class H7VolunteeredPlanTests(unittest.TestCase):
    def test_plan_is_deterministic_complete_and_content_addressed(self) -> None:
        first = build_h7_volunteered_collection_plan(
            _population_rows(),
            source_run=_source_binding(),
        )
        second = build_h7_volunteered_collection_plan(
            _population_rows(),
            source_run=_source_binding(),
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.cases), 6)
        self.assertEqual(len(first.request_bindings), 12)
        self.assertEqual(
            {
                binding.updater_id
                for binding in first.request_bindings
            },
            {BASELINE_UPDATER_ID, MITIGATION_UPDATER_ID},
        )
        self.assertEqual(first.to_dict()["claim_status"], CLAIM_STATUS)
        for schema_name, record in (
            ("h7-volunteered-collection-plan", first.to_dict()),
            (
                "h7-volunteered-request-binding",
                first.request_bindings[0].to_dict(),
            ),
        ):
            schema = SCHEMAS[schema_name]
            self.assertTrue(set(schema["required"]).issubset(record))
            self.assertTrue(set(record).issubset(schema["properties"]))
        with self.assertRaises(ValueError):
            replace(first.cases[0], surface_statement="tampered")

    def test_provider_conversion_is_exact_and_never_imputes(self) -> None:
        plan = build_h7_volunteered_collection_plan(
            _population_rows(),
            source_run=_source_binding(),
        )
        responses, audits = _provider_material(plan)
        updates, evidence = volunteered_updates_from_provider_evidence(
            plan,
            responses,
            audits,
        )
        self.assertEqual(len(updates), len(plan.request_bindings))
        self.assertEqual(len(evidence), len(plan.request_bindings))
        self.assertTrue(
            all(update.directional_log_odds_update > 0.0 for update in updates)
        )
        with self.assertRaises(ValueError):
            volunteered_updates_from_provider_evidence(
                plan,
                responses[:-1],
                audits,
            )
        rejected = list(audits)
        rejected[0] = {**rejected[0], "acceptance_status": "rejected_model_mismatch"}
        with self.assertRaises(ValueError):
            volunteered_updates_from_provider_evidence(
                plan,
                responses,
                rejected,
            )


class H7VolunteeredReviewTests(unittest.TestCase):
    def test_review_is_derived_reproducible_and_source_immutable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _verified_source(root / "runs")
            source = load_verified_h7_source(run_dir)
            plan_dir = root / "plan"
            write_h7_plan_directory(plan_dir, source.plan)
            responses, audits = _provider_material(source.plan)
            responses_path = root / "responses.jsonl"
            audits_path = root / "provider-audit.jsonl"
            _write_jsonl(
                responses_path,
                (response.to_dict() for response in responses),
            )
            _write_jsonl(audits_path, audits)
            source_checksums_before = (run_dir / "SHA256SUMS").read_bytes()

            first = create_h7_volunteered_review(
                source,
                plan_dir,
                responses_path,
                audits_path,
            )
            second = create_h7_volunteered_review(
                source,
                plan_dir,
                responses_path,
                audits_path,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["claim_status"], CLAIM_STATUS)
            review_schema = SCHEMAS["h7-volunteered-review"]
            self.assertTrue(
                set(review_schema["required"]).issubset(first)
            )
            self.assertTrue(
                set(first).issubset(review_schema["properties"])
            )
            evidence_schema = SCHEMAS["h7-volunteered-evidence"]
            evidence_row = first["provider_bound_evidence"][0]
            self.assertTrue(
                set(evidence_schema["required"]).issubset(evidence_row)
            )
            self.assertTrue(
                set(evidence_row).issubset(evidence_schema["properties"])
            )
            self.assertTrue(first["recomputed_h7"]["complete"])
            self.assertIs(first["recomputed_h7"]["criterion_met"], True)
            self.assertFalse(
                first["recomputation_scope"]["source_run_modified"]
            )
            self.assertFalse(
                first["recomputation_scope"]["missing_values_imputed"]
            )
            self.assertEqual(
                (run_dir / "SHA256SUMS").read_bytes(),
                source_checksums_before,
            )

            review_path = root / "review.json"
            review_path.write_text(
                json.dumps(first, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            valid, errors = verify_h7_volunteered_review(
                run_dir,
                plan_dir,
                responses_path,
                audits_path,
                review_path,
            )
            self.assertTrue(valid, errors)
            tampered = {**first, "claim_status": "claimed"}
            review_path.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            valid, _ = verify_h7_volunteered_review(
                run_dir,
                plan_dir,
                responses_path,
                audits_path,
                review_path,
            )
            self.assertFalse(valid)

    def test_cli_plan_review_verify_and_overwrite_refusal(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _verified_source(root / "runs")
            plan_dir = root / "plan"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                first = cli_main(
                    [
                        "control-study",
                        "h7-plan",
                        str(run_dir),
                        str(plan_dir),
                    ]
                )
                with self.assertRaises(SystemExit) as raised:
                    cli_main(
                        [
                            "control-study",
                            "h7-plan",
                            str(run_dir),
                            str(plan_dir),
                        ]
                    )
            self.assertEqual(first, 0)
            self.assertEqual(raised.exception.code, 2)
            self.assertTrue((plan_dir / PLAN_FILENAME).is_file())
            self.assertTrue((plan_dir / BINDINGS_FILENAME).is_file())
            self.assertTrue((plan_dir / REQUESTS_FILENAME).is_file())
            source = load_verified_h7_source(run_dir)
            responses, audits = _provider_material(source.plan)
            responses_path = root / "responses.jsonl"
            audits_path = root / "provider-audit.jsonl"
            review_path = root / "review.json"
            _write_jsonl(
                responses_path,
                (response.to_dict() for response in responses),
            )
            _write_jsonl(audits_path, audits)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                review_status = cli_main(
                    [
                        "control-study",
                        "h7-review",
                        str(run_dir),
                        str(plan_dir),
                        str(responses_path),
                        str(audits_path),
                        str(review_path),
                    ]
                )
                verify_status = cli_main(
                    [
                        "control-study",
                        "h7-verify",
                        str(run_dir),
                        str(plan_dir),
                        str(responses_path),
                        str(audits_path),
                        str(review_path),
                    ]
                )
            self.assertEqual(review_status, 0)
            self.assertEqual(verify_status, 0)
            self.assertTrue(review_path.is_file())

    def test_cli_rejects_input_mutation_before_review_publication(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _verified_source(root / "runs")
            source = load_verified_h7_source(run_dir)
            plan_dir = root / "plan"
            write_h7_plan_directory(plan_dir, source.plan)
            responses, audits = _provider_material(source.plan)
            responses_path = root / "responses.jsonl"
            audits_path = root / "provider-audit.jsonl"
            review_path = root / "review.json"
            _write_jsonl(
                responses_path,
                (response.to_dict() for response in responses),
            )
            _write_jsonl(audits_path, audits)
            snapshots = snapshot_h7_review_inputs(
                plan_dir,
                responses_path,
                audits_path,
            )
            self.assertEqual(
                snapshots.responses.sha256,
                sha256(responses_path.read_bytes()).hexdigest(),
            )

            def mutate_after_review(*args, **kwargs):
                payload = create_h7_volunteered_review(*args, **kwargs)
                audits_path.write_bytes(audits_path.read_bytes() + b"\n")
                return payload

            with (
                patch(
                    "cape_loop.cli.create_h7_volunteered_review",
                    side_effect=mutate_after_review,
                ),
                redirect_stderr(StringIO()),
                redirect_stdout(StringIO()),
                self.assertRaises(SystemExit),
            ):
                cli_main(
                    [
                        "control-study",
                        "h7-review",
                        str(run_dir),
                        str(plan_dir),
                        str(responses_path),
                        str(audits_path),
                        str(review_path),
                    ]
                )
            self.assertFalse(review_path.exists())


if __name__ == "__main__":
    unittest.main()
