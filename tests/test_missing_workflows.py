from __future__ import annotations

from dataclasses import replace
import unittest

from cape_loop.correction_debt import (
    CorrectionProtocol,
    run_correction_debt_experiment,
)
from cape_loop.decoder_study import (
    DecoderTruthLabel,
    ExternalDecoderJudgment,
    ExternalDecoderRequest,
    HumanCollectionRecord,
    analyze_external_decoders,
    analyze_human_evidence_strength,
    fit_decoder_calibration,
    validate_external_decoder_import,
    validate_human_collection,
)
from cape_loop.domains import TRAVEL
from cape_loop.heldout import (
    ParaphraseEvaluationRecord,
    ParaphraseSource,
    TerminalAction,
    build_default_paraphrase_suite,
    build_heldout_terminal_suite,
    evaluate_gate1_paraphrase_transfer,
    generate_paraphrase_cases,
    score_heldout_terminal_actions,
)


class HeldOutParaphraseTests(unittest.TestCase):
    def _cases(self):
        suite = build_default_paraphrase_suite()
        sources = []
        for domain in ("travel", "writing"):
            for mechanism in ("default", "restricted"):
                sources.append(
                    ParaphraseSource.build(
                        source_trial_id=f"{domain}:{mechanism}",
                        domain_id=domain,
                        mechanism=mechanism,
                        selected_option_id=f"{domain}:anchor",
                        selected_label=f"{domain} anchor",
                        selected_ordinal="first",
                        visible_context={
                            "domain": domain,
                            "mechanism": mechanism,
                            "options": ["anchor", "alternative"],
                        },
                    )
                )
        return suite, generate_paraphrase_cases(sources, suite)

    def test_suite_cases_and_gate_criterion_are_bound_and_deterministic(self) -> None:
        suite, cases = self._cases()
        self.assertEqual(
            suite.suite_sha256,
            build_default_paraphrase_suite().suite_sha256,
        )
        self.assertEqual(len(cases), 8)
        self.assertTrue(all(case.split == "test" for case in cases))
        self.assertEqual(
            [case.to_dict() for case in cases],
            [
                case.to_dict()
                for case in generate_paraphrase_cases(
                    [
                        ParaphraseSource.build(
                            source_trial_id=f"{domain}:{mechanism}",
                            domain_id=domain,
                            mechanism=mechanism,
                            selected_option_id=f"{domain}:anchor",
                            selected_label=f"{domain} anchor",
                            selected_ordinal="first",
                            visible_context={
                                "domain": domain,
                                "mechanism": mechanism,
                                "options": ["anchor", "alternative"],
                            },
                        )
                        for domain in ("travel", "writing")
                        for mechanism in ("default", "restricted")
                    ],
                    suite,
                )
            ],
        )

        records = []
        for case in cases:
            records.extend(
                (
                    ParaphraseEvaluationRecord.from_case(
                        case,
                        updater_id="fitted_action_aware",
                        brier=0.10,
                        belief_payload={"kind": "aware", "case": case.case_id},
                    ),
                    ParaphraseEvaluationRecord.from_case(
                        case,
                        updater_id="llm_full_context",
                        brier=0.14,
                        belief_payload={"kind": "llm", "case": case.case_id},
                    ),
                )
            )
        criterion = evaluate_gate1_paraphrase_transfer(
            cases,
            records,
            suite=suite,
            required_mechanisms=2,
        )
        self.assertTrue(criterion.complete)
        self.assertTrue(criterion.verified)
        self.assertEqual(
            criterion.to_dict()["gate_1_argument"],
            True,
        )

        incomplete = evaluate_gate1_paraphrase_transfer(
            cases,
            records[:-1],
            suite=suite,
            required_mechanisms=2,
        )
        self.assertFalse(incomplete.complete)
        self.assertIsNone(incomplete.verified)
        self.assertTrue(incomplete.missing_pairs)

    def test_test_template_leakage_is_rejected(self) -> None:
        suite = build_default_paraphrase_suite()
        test_template = suite.for_split("test")[0]
        with self.assertRaises(ValueError):
            suite.assert_no_test_leakage(
                fitted_template_ids=(test_template.template_id,)
            )
        with self.assertRaises(ValueError):
            suite.assert_no_test_leakage(
                fitted_surface_patterns=(test_template.pattern,)
            )


class HeldOutTerminalTests(unittest.TestCase):
    def test_terminal_suite_is_novel_and_scoring_uses_response_contract(self) -> None:
        suite = build_heldout_terminal_suite(TRAVEL)
        training_ids = {
            option.option_id
            for option in TRAVEL.option_pool + TRAVEL.isolated_options
        }
        training_features = {
            option.features
            for option in TRAVEL.option_pool + TRAVEL.isolated_options
        }
        self.assertTrue(
            training_ids.isdisjoint(
                option.option_id
                for item in suite.items
                for option in item.options
            )
        )
        self.assertTrue(
            training_features.isdisjoint(
                option.features
                for item in suite.items
                for option in item.options
            )
        )

        truth = (2, -1, 1)
        actions = []
        for item in suite.items:
            if item.question_type == "direct_preference_probe":
                assert item.target_attribute is not None
                action = TerminalAction(
                    item_id=item.item_id,
                    item_sha256=item.item_sha256,
                    wording_template_id=item.wording_template_id,
                    question_type=item.question_type,
                    declared_direction=(
                        1 if truth[item.target_attribute] > 0 else -1
                    ),
                )
            else:
                selected = max(
                    item.options,
                    key=lambda option: (
                        sum(
                            coefficient * feature
                            for coefficient, feature in zip(
                                truth, option.features
                            )
                        ),
                        option.option_id,
                    ),
                )
                action = TerminalAction(
                    item_id=item.item_id,
                    item_sha256=item.item_sha256,
                    wording_template_id=item.wording_template_id,
                    question_type=item.question_type,
                    selected_option_id=selected.option_id,
                )
            actions.append(action)
        score = score_heldout_terminal_actions(suite, actions, truth)
        self.assertEqual(score.overall_accuracy, 1.0)
        self.assertEqual(
            set(dict(score.count_by_question_type)),
            {
                "forced_choice",
                "counterfactual_choice",
                "direct_preference_probe",
                "cross_context_choice",
            },
        )

        mismatched = list(actions)
        mismatched[0] = replace(
            mismatched[0],
            wording_template_id="wrong-wording",
        )
        with self.assertRaises(ValueError):
            score_heldout_terminal_actions(suite, mismatched, truth)


def _probabilities_for(theta: tuple[int, int, int], confidence: float):
    rows = []
    for true_value in theta:
        remaining = (1.0 - confidence) / 3.0
        rows.append(
            {
                label: (
                    confidence
                    if int(label) == true_value
                    else remaining
                )
                for label in ("-2", "-1", "+1", "+2")
            }
        )
    return rows


class ExternalDecoderTests(unittest.TestCase):
    def _fixture(self):
        states = (
            ("development-1", "development", (2, -1, 1)),
            ("development-2", "development", (-2, 1, 2)),
            ("test-1", "test", (1, -2, 2)),
            ("test-2", "test", (-1, 2, -2)),
        )
        requests = tuple(
            ExternalDecoderRequest.build(
                request_id=f"request:{state_id}",
                pseudonymous_state_id=state_id,
                representation_id="blinded-native-state-v1",
                evaluation_split=split,
                payload={
                    "persona_summary": f"blinded summary {index}",
                    "attribute_slots": [0, 1, 2],
                },
            )
            for index, (state_id, split, _) in enumerate(states)
        )
        labels = tuple(
            DecoderTruthLabel(state_id, theta, split)
            for state_id, split, theta in states
        )
        judgment_rows = []
        for request, (_, _, theta) in zip(requests, states):
            for family, confidence in (
                ("decoder-family-a", 0.70),
                ("decoder-family-b", 0.60),
            ):
                judgment_rows.append(
                    ExternalDecoderJudgment.parse(
                        {
                            "schema_version": 1,
                            "request_id": request.request_id,
                            "request_sha256": request.request_sha256,
                            "decoder_instance_id": f"{family}:instance-1",
                            "decoder_family_id": family,
                            "judgment_origin": "external_model",
                            "source_descriptor": f"{family}:frozen-model",
                            "blind_to_system_identity": True,
                            "blind_to_latent_truth": True,
                            "probabilities": _probabilities_for(
                                theta, confidence
                            ),
                        }
                    )
                )
        return requests, tuple(judgment_rows), labels

    def test_external_judgments_are_validated_calibrated_and_compared(self) -> None:
        requests, judgments, labels = self._fixture()
        audit = validate_external_decoder_import(requests, judgments)
        self.assertTrue(audit.complete_coverage)
        self.assertTrue(audit.source_design_eligible)
        self.assertIn("not proof", audit.caveat)

        calibration = fit_decoder_calibration(
            requests, judgments, labels
        )
        self.assertEqual(
            {family for family, _ in calibration.calibrators},
            {"decoder-family-a", "decoder-family-b"},
        )
        self.assertTrue(
            all(
                calibrator.fitted_splits == ("development",)
                for _, calibrator in calibration.calibrators
            )
        )
        analysis = analyze_external_decoders(
            requests,
            judgments,
            labels,
            calibration=calibration,
            reliability_bins=5,
        )
        self.assertEqual(len(analysis.family_metrics), 2)
        self.assertEqual(len(analysis.agreement), 1)
        self.assertEqual(
            analysis.agreement[0].shared_request_count,
            2,
        )
        self.assertIn(
            "deterministic native projections are excluded",
            analysis.interpretation_boundary,
        )

    def test_empty_decoder_import_is_not_eligible(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires at least one request",
        ):
            validate_external_decoder_import((), ())

    def test_decoder_payload_leakage_and_deterministic_view_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExternalDecoderRequest.build(
                request_id="leaky",
                pseudonymous_state_id="state",
                representation_id="view",
                evaluation_split="test",
                payload={"updater_id": "secret"},
            )
        requests, judgments, _ = self._fixture()
        malformed = judgments[0].to_dict()
        malformed["judgment_origin"] = "deterministic_view"
        with self.assertRaises(ValueError):
            ExternalDecoderJudgment.parse(malformed)
        unblinded = replace(judgments[0], blind_to_latent_truth=False)
        with self.assertRaises(ValueError):
            validate_external_decoder_import(
                requests,
                (unblinded,) + judgments[1:],
            )


class HumanCollectionTests(unittest.TestCase):
    def _fixture(self):
        codebook = {"assignment-1": {}}
        ratings = {
            "volunteered": (7, 6),
            "balanced": (6, 5),
            "restricted": (2, 2),
            "default": (3, 2),
            "suggested": (4, 3),
        }
        records = []
        for condition_index, (condition, values) in enumerate(ratings.items()):
            display_id = f"item-{condition_index + 1:04d}"
            codebook["assignment-1"][display_id] = {
                "item_id": f"source-{condition}",
                "scenario_id": "matched-scenario",
                "condition": condition,
            }
            for participant_index, rating in enumerate(values, start=1):
                records.append(
                    HumanCollectionRecord(
                        participant_code=f"participant-{participant_index}",
                        assignment_id="assignment-1",
                        assignment_protocol_id="assignment-v1",
                        display_id=display_id,
                        rating=rating,
                        response_time_ms=1000 + 100 * condition_index,
                        consent_version="consent-v1",
                        consented=True,
                        blinding_version="blind-v1",
                        comprehension_check_id="check-v1",
                        comprehension_passed=True,
                    )
                )
        records.append(
            HumanCollectionRecord(
                participant_code="participant-no-consent",
                assignment_id="assignment-1",
                assignment_protocol_id="assignment-v1",
                display_id="item-0001",
                rating=7,
                response_time_ms=900,
                consent_version="consent-v1",
                consented=False,
                blinding_version="blind-v1",
                comprehension_check_id="check-v1",
                comprehension_passed=True,
            )
        )
        return tuple(records), codebook

    def test_human_import_excludes_ineligible_rows_and_reports_paired_order(self) -> None:
        records, codebook = self._fixture()
        audit = validate_human_collection(
            records,
            assignment_codebooks=codebook,
            expected_assignment_protocol_id="assignment-v1",
            expected_consent_version="consent-v1",
            expected_blinding_version="blind-v1",
        )
        self.assertEqual(audit.excluded_no_consent, 1)
        self.assertEqual(audit.analysis_eligible_count, 10)
        self.assertIn("does not assert", audit.ethics_boundary)

        analysis = analyze_human_evidence_strength(
            records,
            assignment_codebooks=codebook,
            expected_assignment_protocol_id="assignment-v1",
            expected_consent_version="consent-v1",
            expected_blinding_version="blind-v1",
            bootstrap_replicates=50,
            seed=9,
        )
        self.assertEqual(
            analysis.evidence_strength_ranking[0],
            "volunteered",
        )
        contrasts = {
            item.contrast_id: item for item in analysis.paired_contrasts
        }
        self.assertGreater(
            contrasts["balanced-minus-restricted"].mean_difference,
            0.0,
        )
        self.assertIn(
            "no ethics-review approval is inferred",
            analysis.interpretation_boundary,
        )

    def test_human_schema_does_not_accept_unblinded_condition(self) -> None:
        records, _ = self._fixture()
        raw = records[0].to_dict()
        raw["condition"] = "volunteered"
        with self.assertRaises(ValueError):
            HumanCollectionRecord.parse(raw)


class CorrectionDebtTests(unittest.TestCase):
    def test_stage_gate_pairing_and_reference_debt_are_deterministic(self) -> None:
        with self.assertRaises(ValueError):
            run_correction_debt_experiment(
                pair_truth_directions={"pair-1": 1},
                stage_gate_authorized=False,
            )
        protocol = CorrectionProtocol(max_balanced_turns=8)
        first = run_correction_debt_experiment(
            pair_truth_directions={"pair-1": 1, "pair-2": -1},
            stage_gate_authorized=True,
            protocol=protocol,
        )
        second = run_correction_debt_experiment(
            pair_truth_directions={"pair-2": -1, "pair-1": 1},
            stage_gate_authorized=True,
            protocol=protocol,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(len(first.arms), 16)
        self.assertEqual(len(first.paired_debts), 8)
        self.assertEqual(first.claim_status, "not_claimed")
        self.assertTrue(
            all(
                row.recovery_error_auc_debt > 0.0
                for row in first.paired_debts
            )
        )
        self.assertTrue(
            all(
                row.persistent_wrong_memory_debt > 0
                for row in first.paired_debts
            )
        )
        for pair in ("pair-1", "pair-2"):
            for stage in protocol.stages:
                false = next(
                    arm
                    for arm in first.arms
                    if arm.pair_id == pair
                    and arm.stage_id == stage.stage_id
                    and arm.seed_condition == "false"
                )
                correct = next(
                    arm
                    for arm in first.arms
                    if arm.pair_id == pair
                    and arm.stage_id == stage.stage_id
                    and arm.seed_condition == "correct"
                )
                self.assertEqual(
                    [
                        item.common_evidence_key
                        for item in false.snapshots
                    ],
                    [
                        item.common_evidence_key
                        for item in correct.snapshots
                    ],
                )


if __name__ == "__main__":
    unittest.main()
