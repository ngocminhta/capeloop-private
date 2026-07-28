from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

from cape_loop.artifacts import RunArtifacts, canonical_json
from cape_loop.beliefs import MarginalPreferenceBelief, PreferenceBelief
from cape_loop.cli import build_parser
from cape_loop.config import (
    AppConfig,
    ArtifactSection,
    ExperimentSection,
    InferenceSection,
    LLMSection,
    RunSection,
    SensitivitySection,
)
from cape_loop.gates import GateCriterion, GateReport
from cape_loop.heldout import (
    ParaphraseEvaluationRecord,
    ParaphraseSource,
    build_default_paraphrase_suite,
    evaluate_gate1_paraphrase_transfer,
    generate_paraphrase_cases,
)
from cape_loop.llm_exchange import ATTRIBUTES, VALUES, LLMRequest, LLMResponse
from cape_loop.provider_attempts import DurableProviderAttemptLedger
from cape_loop.robustness_review import (
    CRITERION_IDS,
    build_gate6_cross_run_review,
    verify_gate6_cross_run_review,
)
from cape_loop.sensitivity import sensitivity_grid


def _digest(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _response(request: LLMRequest, model: str) -> LLMResponse:
    return LLMResponse.parse(
        {
            "schema_version": 1,
            "request_id": request.request_id,
            "prompt_sha256": request.prompt_sha256,
            "model_id": model,
            "beliefs": {
                attribute: {value: 0.25 for value in VALUES}
                for attribute in ATTRIBUTES
            },
            "raw_response_sha256": _digest(
                {"request_id": request.request_id, "model": model}
            ),
        }
    )


def _write_provider_exchange(
    run: RunArtifacts,
    *,
    requests: tuple[LLMRequest, ...],
    responses: tuple[LLMResponse, ...],
    model: str,
) -> None:
    run.write_jsonl("llm/requests.jsonl", (row.to_dict() for row in requests))
    run.write_jsonl(
        "llm/responses.jsonl",
        (row.to_dict() for row in responses),
    )
    run.write_json(
        "llm/exchange-manifest.json",
        {
            "schema_version": 1,
            "prompts_retained": True,
            "requests": [
                {
                    "request_id": request.request_id,
                    "updater_id": request.updater_id,
                    "view": request.view,
                    "prompt_sha256": request.prompt_sha256,
                }
                for request in requests
            ],
            "models": [model],
            "execution_mode": "openai",
            "probability_calibration": "none",
        },
    )
    run.write_json(
        "models/llm-calibration.json",
        {
            "schema_version": 1,
            "kind": "none",
            "fitted_split": None,
            "calibrators": {},
            "test_labels_used": False,
        },
    )

    attempts_path = run.path / "llm" / "transport-attempts.jsonl"
    ledger = DurableProviderAttemptLedger(
        attempts_path,
        provider_name="openai",
        model_requested=model,
    )
    audit_rows = []
    response_by_id = {row.request_id: row for row in responses}
    for ordinal, request in enumerate(requests, start=1):
        response = response_by_id[request.request_id]
        body_sha256 = _digest({"body": request.request_id})
        audit = {
            "schema_version": 1,
            "provider": "openai",
            "acceptance_status": "accepted",
            "request_id": request.request_id,
            "prompt_sha256": request.prompt_sha256,
            "model_requested": model,
            "model_returned": model,
            "raw_response_sha256": response.raw_response_sha256,
            "replay_response": response.to_dict(),
        }
        prepared = SimpleNamespace(
            endpoint="https://api.openai.com/v1/responses",
            body_sha256=body_sha256,
            client_request_id=f"fixture-{ordinal}",
            idempotency_key=f"fixture-idempotency-{ordinal}",
            estimated_max_tokens=100,
        )
        attempt_id = ledger.start(
            request,
            prepared,
            started_at=f"2026-07-27T00:00:{ordinal:02d}Z",
        )
        ledger.settle(
            attempt_id,
            settled_at=f"2026-07-27T00:01:{ordinal:02d}Z",
            outcome="success",
            automatic_retry_safe=False,
            http_status=200,
            charged_tokens=10,
            server_request_id=f"server-{ordinal}",
            response_body_sha256=response.raw_response_sha256,
            response_record=response.to_dict(),
            provider_audit=audit,
        )
        audit_rows.append(audit)
    audit_path = run.write_jsonl("llm/provider-audit.jsonl", audit_rows)
    run.write_json(
        "llm/provider-manifest.json",
        {
            "schema_version": 1,
            "provider": "openai",
            "model_requested": model,
            "reasoning_effort": "low",
            "requests_used": len(requests),
            "requests_executed": len(requests),
            "requests_resumed": 0,
            "total_tokens": 10 * len(requests),
            "transport_attempt_count": len(requests),
            "request_budget_unit": "physical_http_attempt",
            "provider_audit_file": "llm/provider-audit.jsonl",
            "provider_audit_sha256": sha256(
                audit_path.read_bytes()
            ).hexdigest(),
            "transport_attempts_file": "llm/transport-attempts.jsonl",
            "transport_attempts_sha256": sha256(
                attempts_path.read_bytes()
            ).hexdigest(),
            "transport_attempt_event_count": 2 * len(requests),
            "external_recovery_journal_retained": True,
            "credentials_retained": False,
        },
    )


def _base_llm(model: str) -> LLMSection:
    return LLMSection(
        mode="openai",
        calibration="none",
        model="",
        reasoning_effort="low",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com",
        max_retries=0,
        max_output_tokens=128,
        max_requests=100,
        max_total_tokens=100_000,
    )


def _sensitivity_config(
    *,
    name: str,
    seed: int,
    model: str,
    suggestion_threshold: float = 0.20,
) -> AppConfig:
    return AppConfig(
        run=RunSection(name=name, seed=seed, deterministic=False),
        experiment=ExperimentSection(
            kind="sensitivity",
            domains=("travel", "writing"),
            mechanisms=("ranking", "default", "suggestion"),
            response_modes=("naturally_sampled",),
            prior_strengths=(0.0,),
            policies=("balanced", "soft_profile_conditioned"),
            updaters=(
                "fitted_action_aware",
                "fitted_action_unaware",
                "llm_full_context",
            ),
            users=1,
            trajectories_per_cell=1,
            turns=1,
            bootstrap_replicates=0,
        ),
        inference=InferenceSection(
            training_interactions=16,
            fit_steps=5,
            learning_rate=0.03,
            l2=0.001,
            calibration="none",
        ),
        sensitivity=SensitivitySection(
            decision_noise_values=(1.0,),
            presentation_multipliers=(1.0,),
            rank_multipliers=(1.0,),
            default_multipliers=(1.0,),
            suggestion_multipliers=(1.0,),
            profile_strength_values=(0.8,),
            prior_uncertainty_values=(0.0,),
            trajectory_lengths=(3,),
            response_model_families=("random_utility",),
            rule_noise_values=(0.15,),
            phase_min_suggestion_rejection_rate=suggestion_threshold,
        ),
        llm=_base_llm(model),
        artifacts=ArtifactSection(
            retain_events=True,
            retain_prompts=True,
            checksum_manifest=True,
        ),
    ).validated()


def _experiment_a_config(*, name: str, seed: int, model: str) -> AppConfig:
    return AppConfig(
        run=RunSection(name=name, seed=seed, deterministic=False),
        experiment=ExperimentSection(
            kind="provenance_audit",
            domains=("travel", "writing"),
            mechanisms=("balanced", "restricted", "default", "suggested"),
            response_modes=("controlled_anchor", "naturally_sampled"),
            prior_strengths=(0.0,),
            policies=("balanced",),
            updaters=(
                "fitted_action_aware",
                "fitted_action_unaware",
                "llm_full_context",
            ),
            users=1,
            trajectories_per_cell=1,
            turns=1,
            bootstrap_replicates=0,
        ),
        inference=InferenceSection(
            training_interactions=16,
            fit_steps=5,
            learning_rate=0.03,
            l2=0.001,
            calibration="none",
        ),
        llm=_base_llm(model),
        artifacts=ArtifactSection(
            retain_events=True,
            retain_prompts=True,
            checksum_manifest=True,
        ),
    ).validated()


def _make_sensitivity_run(
    root: Path,
    *,
    family: str,
    model: str,
    seed: int,
    suggestion_threshold: float = 0.20,
) -> Path:
    config = _sensitivity_config(
        name=f"sensitivity-{family}",
        seed=seed,
        model=model,
        suggestion_threshold=suggestion_threshold,
    )
    run = RunArtifacts.create(config, root=root)
    request = LLMRequest.build(
        request_id=f"{family}-sensitivity",
        updater_id="llm_full_context",
        view="full_context",
        prior={"fixture": "uniform"},
        observation={"selected_option": "option-a", "surface_response": "A"},
        context={"domain": "travel", "options": ["option-a", "option-b"]},
    )
    response = _response(request, model)
    _write_provider_exchange(
        run,
        requests=(request,),
        responses=(response,),
        model=model,
    )
    point = sensitivity_grid(
        design=config.sensitivity.design,
        decision_noise_values=config.sensitivity.decision_noise_values,
        presentation_multipliers=config.sensitivity.presentation_multipliers,
        rank_multipliers=config.sensitivity.rank_multipliers,
        default_multipliers=config.sensitivity.default_multipliers,
        suggestion_multipliers=config.sensitivity.suggestion_multipliers,
        profile_strength_values=config.sensitivity.profile_strength_values,
        prior_uncertainty_values=config.sensitivity.prior_uncertainty_values,
        trajectory_lengths=config.sensitivity.trajectory_lengths,
        response_model_families=config.sensitivity.response_model_families,
        rule_noise_values=config.sensitivity.rule_noise_values,
    )[0]
    phase = {
        "schema_version": 1,
        **point.to_dict(),
        "phase_target_updater_id": "llm_full_context",
        "phase_target_is_llm": True,
        "phase_target_is_live_llm": True,
        "llm_execution_mode": "openai",
        "criteria_complete": True,
        "operational_joint_region": True,
    }
    run.write_jsonl("metrics/sensitivity-phase-points.jsonl", (phase,))
    run.write_jsonl(
        "metrics/sensitivity-phase-domains.jsonl",
        (
            {
                **phase,
                "domain_id": domain,
            }
            for domain in ("travel", "writing")
        ),
    )
    run.write_jsonl(
        "models/sensitivity-fits.jsonl",
        (
            {
                "schema_version": 1,
                **point.to_dict(),
                "raw_fitted_models": {"aware": {"fixture": True}},
                "fitted_models": {"aware": {"fixture": True}},
            },
        ),
    )
    gate = GateReport(
        gate_id="gate-6",
        title="Robustness",
        criteria=(
            GateCriterion(CRITERION_IDS[0], "another", False, {}, "fixture"),
            GateCriterion(CRITERION_IDS[1], "broad", False, {}, "fixture"),
            GateCriterion(CRITERION_IDS[2], "domains", True, {}, "fixture"),
            GateCriterion(CRITERION_IDS[3], "families", None, {}, "fixture"),
            GateCriterion(CRITERION_IDS[4], "paraphrases", None, {}, "fixture"),
            GateCriterion(CRITERION_IDS[5], "references", True, {}, "fixture"),
        ),
    )
    run.write_json(
        "metrics/gate-report.json",
        {
            "schema_version": 1,
            "claim_status": "not_claimed",
            "gates": [gate.to_dict()],
        },
    )
    run.finalize(
        {
            "experiment": "sensitivity",
            "scientific_claim_status": "not_claimed",
            "declared_points": 1,
            "completed_points": 1,
        }
    )
    return run.path


def _make_experiment_a_run(
    root: Path,
    *,
    family: str,
    model: str,
    seed: int,
) -> Path:
    config = _experiment_a_config(
        name=f"experiment-a-{family}",
        seed=seed,
        model=model,
    )
    run = RunArtifacts.create(config, root=root)
    suite = build_default_paraphrase_suite()
    sources = []
    contexts = {}
    for domain in ("travel", "writing"):
        for mechanism in ("default", "restricted"):
            source_id = f"{domain}:{mechanism}"
            context = {
                "domain": domain,
                "mechanism": mechanism,
                "options": ["anchor", "alternative"],
            }
            sources.append(
                ParaphraseSource.build(
                    source_trial_id=source_id,
                    domain_id=domain,
                    mechanism=mechanism,
                    selected_option_id="anchor",
                    selected_label=f"{domain} anchor",
                    selected_ordinal="first",
                    visible_context=context,
                )
            )
            contexts[source_id] = context
    cases = generate_paraphrase_cases(sources, suite)
    requests = tuple(
        LLMRequest.build(
            request_id=f"{family}:{case.case_id}",
            updater_id="llm_full_context",
            view="full_context",
            prior={"fixture": "uniform"},
            observation={
                "selected_option": case.selected_option_id,
                "surface_response": case.surface_response,
            },
            context=contexts[case.source_trial_id],
        )
        for case in cases
    )
    responses = tuple(_response(request, model) for request in requests)
    _write_provider_exchange(
        run,
        requests=requests,
        responses=responses,
        model=model,
    )
    response_by_id = {row.request_id: row for row in responses}
    request_by_case = dict(zip((case.case_id for case in cases), requests))
    records = []
    for case in cases:
        response = response_by_id[request_by_case[case.case_id].request_id]
        rows = tuple(
            tuple(
                float(response.beliefs[attribute][value])
                for value in VALUES
            )
            for attribute in ATTRIBUTES
        )
        belief = PreferenceBelief.from_marginals(
            MarginalPreferenceBelief(rows)  # type: ignore[arg-type]
        )
        records.extend(
            (
                ParaphraseEvaluationRecord.from_case(
                    case,
                    updater_id="fitted_action_aware",
                    brier=0.10,
                    belief_payload={"fixture": "aware", "case": case.case_id},
                ),
                ParaphraseEvaluationRecord.from_case(
                    case,
                    updater_id="llm_full_context",
                    brier=0.14,
                    belief_payload=belief.to_dict(),
                ),
            )
        )
    criterion = evaluate_gate1_paraphrase_transfer(
        cases,
        records,
        suite=suite,
        required_mechanisms=2,
        required_domains=("travel", "writing"),
    )
    run.write_json("models/held-out-paraphrase-suite.json", suite.to_dict())
    run.write_jsonl(
        "events/experiment-a-held-out-paraphrases.jsonl",
        (case.to_dict() for case in cases),
    )
    run.write_jsonl(
        "metrics/experiment-a-held-out-paraphrase-scores.jsonl",
        (record.to_dict() for record in records),
    )
    run.write_json(
        "metrics/experiment-a-held-out-paraphrase-transfer.json",
        criterion.to_dict(),
    )
    run.write_json(
        "metrics/gate-report.json",
        {
            "schema_version": 1,
            "claim_status": "not_claimed",
            "gates": [
                {
                    "schema_version": 1,
                    "gate_id": "gate-1",
                    "claim_status": "not_claimed",
                }
            ],
        },
    )
    run.finalize(
        {
            "experiment": "A",
            "scientific_claim_status": "not_claimed",
        }
    )
    return run.path


class Gate6CrossRunReviewTests(unittest.TestCase):
    def _study(self, root: Path, *, mismatched_grid: bool = False):
        pairs = []
        for index, (family, model) in enumerate(
            (("family-alpha", "model-alpha"), ("family-beta", "model-beta"))
        ):
            sensitivity = _make_sensitivity_run(
                root,
                family=family,
                model=model,
                seed=100 + index,
                suggestion_threshold=(
                    0.25 if mismatched_grid and index == 1 else 0.20
                ),
            )
            experiment_a = _make_experiment_a_run(
                root,
                family=family,
                model=model,
                seed=200 + index,
            )
            pairs.append(
                {
                    "pair_id": f"pair-{index + 1}",
                    "family_id": family,
                    "sensitivity_run": {
                        "path": str(sensitivity),
                        "run_id": sensitivity.name,
                        "sha256sums_sha256": sha256(
                            (sensitivity / "SHA256SUMS").read_bytes()
                        ).hexdigest(),
                    },
                    "experiment_a_run": {
                        "path": str(experiment_a),
                        "run_id": experiment_a.name,
                        "sha256sums_sha256": sha256(
                            (experiment_a / "SHA256SUMS").read_bytes()
                        ).hexdigest(),
                    },
                    "model_binding": {
                        "provider_id": "openai",
                        "provider_source_id": "openai-first-party-responses",
                        "requested_model_id": model,
                        "response_model_id": model,
                        "upstream_provider_id": None,
                        "upstream_model_id": None,
                    },
                }
            )
        declaration = {
            "schema_version": 1,
            "artifact_kind": "gate6-cross-run-declaration",
            "declaration_id": "fixture-gate6-declaration",
            "review_authority": {
                "responsible_researcher_id": "fixture-researcher",
                "reviewed_at_utc": "2026-07-27T12:00:00Z",
                "preregistration_reference": "fixture-preregistration",
                "family_assignments_declared_before_outcome_review": True,
                "source_identities_reviewed": True,
            },
            "statistical_independence_claimed": False,
            "pairs": pairs,
        }
        declaration_path = root / "gate6-declaration.json"
        declaration_path.write_text(
            canonical_json(declaration) + "\n",
            encoding="utf-8",
        )
        return declaration_path, pairs

    def test_builds_and_reverifies_six_clause_review(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declaration, _ = self._study(root)
            output = root / "gate6-review"
            result = build_gate6_cross_run_review(
                declaration_path=declaration,
                output_dir=output,
            )
            self.assertEqual(result["claim_status"], "not_claimed")
            valid, errors = verify_gate6_cross_run_review(
                output,
                reverify_sources=True,
            )
            self.assertTrue(valid, errors)
            gate = json.loads(
                (output / "metrics/gate-6.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row["criterion_id"] for row in gate["criteria"]],
                list(CRITERION_IDS),
            )
            self.assertFalse(gate["criteria"][0]["passed"])
            self.assertFalse(gate["criteria"][1]["passed"])
            self.assertTrue(gate["criteria"][2]["passed"])
            self.assertTrue(gate["criteria"][3]["passed"])
            self.assertTrue(gate["criteria"][4]["passed"])
            self.assertTrue(gate["criteria"][5]["passed"])
            self.assertEqual(gate["claim_status"], "not_claimed")

    def test_cli_exposes_build_and_source_reverification(self) -> None:
        parser = build_parser()
        build = parser.parse_args(
            ["gate6-review", "build", "declaration.json", "review"]
        )
        self.assertEqual(build.gate6_review_command, "build")
        verify = parser.parse_args(
            ["gate6-review", "verify", "review", "--reverify-sources"]
        )
        self.assertTrue(verify.reverify_sources)

    def test_rejects_scientifically_mismatched_sensitivity_grids(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declaration, _ = self._study(root, mismatched_grid=True)
            with self.assertRaisesRegex(
                ValueError,
                "scientific grids/configurations differ",
            ):
                build_gate6_cross_run_review(
                    declaration_path=declaration,
                    output_dir=root / "review",
                )

    def test_checksum_tamper_and_source_tamper_are_detected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declaration, pairs = self._study(root)
            output = root / "review"
            build_gate6_cross_run_review(
                declaration_path=declaration,
                output_dir=output,
            )
            gate_path = output / "metrics/gate-6.json"
            gate_path.write_text("{}\n", encoding="utf-8")
            valid, errors = verify_gate6_cross_run_review(output)
            self.assertFalse(valid)
            self.assertTrue(
                any("checksum mismatch" in error for error in errors),
                errors,
            )

            output_two = root / "review-two"
            build_gate6_cross_run_review(
                declaration_path=declaration,
                output_dir=output_two,
            )
            source = Path(pairs[0]["sensitivity_run"]["path"])
            source_gate = source / "metrics/gate-report.json"
            source_gate.write_text("{}\n", encoding="utf-8")
            valid, errors = verify_gate6_cross_run_review(
                output_two,
                reverify_sources=True,
            )
            self.assertFalse(valid)
            self.assertTrue(
                any("source run verification failed" in error for error in errors),
                errors,
            )

    def test_standalone_verify_does_not_require_source_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declaration, pairs = self._study(root)
            output = root / "review"
            build_gate6_cross_run_review(
                declaration_path=declaration,
                output_dir=output,
            )
            for index, pair in enumerate(pairs):
                for key in ("sensitivity_run", "experiment_a_run"):
                    source = Path(pair[key]["path"])
                    source.rename(root / f"moved-{index}-{key}")
            valid, errors = verify_gate6_cross_run_review(output)
            self.assertTrue(valid, errors)
            valid, errors = verify_gate6_cross_run_review(
                output,
                reverify_sources=True,
            )
            self.assertFalse(valid)

    def test_strict_declaration_and_no_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declaration, _ = self._study(root)
            raw = json.loads(declaration.read_text(encoding="utf-8"))
            raw["unknown"] = True
            declaration.write_text(
                canonical_json(raw) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                build_gate6_cross_run_review(
                    declaration_path=declaration,
                    output_dir=root / "review",
                )

            declaration, _ = self._study(root / "second")
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                build_gate6_cross_run_review(
                    declaration_path=declaration,
                    output_dir=output,
                )

    def test_sibling_lock_prevents_concurrent_publication(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declaration, _ = self._study(root)
            output = root / "review"
            lock = root / ".review.gate6-review.lock"
            lock.write_bytes(b"held")
            with self.assertRaisesRegex(FileExistsError, "is locked"):
                build_gate6_cross_run_review(
                    declaration_path=declaration,
                    output_dir=output,
                )
            self.assertFalse(output.exists())
            self.assertEqual(lock.read_bytes(), b"held")


if __name__ == "__main__":
    unittest.main()
