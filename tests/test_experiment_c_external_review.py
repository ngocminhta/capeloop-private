from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import unittest
from unittest.mock import patch

import cape_loop.experiment_c_review as review_module
from cape_loop.artifacts import verify_run
from cape_loop.cli import build_parser
from cape_loop.config import (
    AppConfig,
    ExperimentSection,
    InferenceSection,
    RunSection,
)
from cape_loop.decoder_study import (
    ExternalDecoderJudgment,
    read_external_decoder_requests,
)
from cape_loop.experiment_c_review import (
    PACKET_CODEBOOK,
    PACKET_MANIFEST,
    PACKET_REQUESTS,
    RESCORE_BASIS,
    import_experiment_c_external_rescore,
    verify_experiment_c_external_rescore,
)
from cape_loop.gate_review import (
    DIRECT_FIRST_PARTY_COLLECTION_PROVENANCE,
    OPENROUTER_COLLECTION_PROVENANCE,
)
from cape_loop.runner import run_experiment
from cape_loop.schema_export import SCHEMAS


class ExperimentCExternalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        config = AppConfig(
            run=RunSection(name="experiment-c-external-review-test", seed=47),
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
                updaters=("response_only", "episodic_memory"),
                users=1,
                trajectories_per_cell=1,
                turns=2,
                bootstrap_replicates=20,
            ),
            inference=InferenceSection(
                training_interactions=24,
                fit_steps=10,
                learning_rate=0.04,
                l2=0.001,
            ),
        )
        result = run_experiment(config, output_root=cls.root / "runs")
        cls.run_dir = Path(result["run_dir"])
        cls.requests = read_external_decoder_requests(
            cls.run_dir / PACKET_REQUESTS
        )
        first = (
            (0.55, 0.15, 0.15, 0.15),
            (0.20, 0.30, 0.30, 0.20),
            (0.10, 0.20, 0.30, 0.40),
        )
        second = (
            (0.45, 0.20, 0.20, 0.15),
            (0.15, 0.35, 0.25, 0.25),
            (0.20, 0.10, 0.40, 0.30),
        )
        cls.judgments = tuple(
            ExternalDecoderJudgment(
                request_id=request.request_id,
                request_sha256=request.request_sha256,
                decoder_instance_id=instance,
                decoder_family_id=family,
                judgment_origin="external_model",
                source_descriptor=descriptor,
                blind_to_system_identity=True,
                blind_to_latent_truth=True,
                probabilities=probabilities,
            )
            for request in cls.requests
            for instance, family, descriptor, probabilities in (
                ("decoder-a", "family-a", "source-a", first),
                ("decoder-b", "family-b", "source-b", second),
            )
        )
        cls.judgments_path = cls.root / "judgments.jsonl"
        cls._write_judgments(cls.judgments_path, cls.judgments)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @staticmethod
    def _write_judgments(
        path: Path,
        judgments: tuple[ExternalDecoderJudgment, ...],
    ) -> None:
        path.write_text(
            "".join(
                json.dumps(
                    judgment.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for judgment in judgments
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _jsonl(path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _assert_no_partial_output(self, output: Path) -> None:
        self.assertFalse(output.exists())
        self.assertFalse(output.is_symlink())
        self.assertFalse(
            (
                output.parent
                / f".{output.name}.external-rescore.lock"
            ).exists()
        )
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.*.staging")),
            [],
        )

    def test_runner_exports_complete_blinded_fixed_and_endogenous_packet(
        self,
    ) -> None:
        valid, errors = verify_run(self.run_dir)
        self.assertTrue(valid, errors)
        self.assertEqual(len(self.requests), 6)
        self.assertEqual(
            {request.evaluation_split for request in self.requests},
            {"development", "test"},
        )
        serialized_payloads = json.dumps(
            [request.payload for request in self.requests],
            sort_keys=True,
        )
        for protected in (
            "user_id",
            "updater_id",
            "latent_truth",
            "memory_kind",
        ):
            self.assertNotIn(f'"{protected}"', serialized_payloads)
        codebook = self._jsonl(self.run_dir / PACKET_CODEBOOK)
        self.assertEqual(len(codebook), len(self.requests))
        self.assertEqual(
            {row["regime"] for row in codebook},
            {
                "fixed_balanced",
                "fixed_biased",
                "endogenous_closed_loop",
            },
        )
        self.assertEqual(
            {
                row["source_state_file"]
                for row in codebook
            },
            {
                "events/experiment-c-replays.jsonl",
                "events/experiment-c-endogenous.jsonl",
            },
        )
        manifest = json.loads(
            (self.run_dir / PACKET_MANIFEST).read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["status"], "ready_for_external_judgments"
        )
        self.assertTrue(manifest["one_request_per_native_metric_row"])
        self.assertFalse(
            manifest["metadata_eligibility_is_statistical_independence"]
        )

    def test_import_rescores_only_native_rows_and_is_verifiable(self) -> None:
        output = self.root / "review-success"
        result = import_experiment_c_external_rescore(
            run_dir=self.run_dir,
            judgments_path=self.judgments_path,
            output_dir=output,
        )
        self.assertEqual(result["claim_status"], "not_claimed")
        self.assertEqual(result["native_row_count"], 6)
        self.assertEqual(result["external_score_count"], 12)
        valid, errors = verify_experiment_c_external_rescore(
            output,
            source_run_dir=self.run_dir,
        )
        self.assertTrue(valid, errors)

        source = self._jsonl(
            self.run_dir / "metrics/experiment-c.jsonl"
        )
        rescored = self._jsonl(
            output / "metrics/experiment-c-rescored.jsonl"
        )
        self.assertEqual(len(source), len(rescored))
        for original, changed in zip(source, rescored):
            if original["updater_id"] == "response_only":
                self.assertEqual(original, changed)
            else:
                self.assertEqual(changed["score_basis"], RESCORE_BASIS)
                self.assertEqual(changed["predicted_option_ids"], [])
                protected = set(original) - {
                    "profile_error",
                    "behavioral_accuracy",
                    "cross_context_accuracy",
                    "intrinsic_regret",
                    "predicted_option_ids",
                    "score_basis",
                    "ranking_score",
                }
                self.assertTrue(
                    all(original[key] == changed[key] for key in protected)
                )
        calibration = json.loads(
            (output / "metrics/calibration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(calibration["fitted_split"], "development")
        self.assertTrue(
            all(
                row["fitted_splits"] == ["development"]
                for row in calibration["calibrators"].values()
            )
        )
        review = json.loads(
            (output / "review.json").read_text(encoding="utf-8")
        )
        self.assertFalse(
            review["validation"]["test_labels_used_for_calibration"]
        )
        self.assertFalse(
            review["validation"]["source_design"][
                "statistical_independence_claimed"
            ]
        )
        source_design = review["validation"]["source_design"]
        self.assertEqual(
            source_design["provenance_mode"],
            "reviewed_generic_judgments",
        )
        self.assertFalse(source_design["provider_provenance_validated"])
        self.assertTrue(
            source_design["caller_declared_source_metadata_only"]
        )
        self.assertIsNone(source_design["official_collection_inputs"])

    def test_official_collection_provenance_is_validated_twice(self) -> None:
        collection = self.root / "mock-official-collection"
        collection.mkdir()
        output = self.root / "review-official-provenance"
        inputs = {
            "decoder_judgments": {
                "filename": "judgments.jsonl",
                "sha256": "a" * 64,
                "bytes": self.judgments_path.stat().st_size,
                "record_count": len(self.judgments),
            }
        }
        summary = {
            "provenance_mode": "validated_direct_first_party_collection",
            "collection_status": "complete",
            "providers": ["anthropic", "google_gemini"],
        }
        with patch.object(
            review_module,
            "validate_selected_external_decoder_collection",
            return_value=(self.judgments, inputs, summary),
        ) as validate:
            result = import_experiment_c_external_rescore(
                run_dir=self.run_dir,
                judgments_path=self.judgments_path,
                output_dir=output,
                external_collection_dir=collection,
                allow_reviewed_generic_decoders=False,
            )
        self.assertEqual(validate.call_count, 2)
        self.assertTrue(result["provider_provenance_validated"])
        review = json.loads(
            (output / "review.json").read_text(encoding="utf-8")
        )
        design = review["validation"]["source_design"]
        self.assertEqual(
            design["provenance_mode"],
            "validated_direct_first_party_collection",
        )
        self.assertTrue(design["provider_provenance_validated"])
        self.assertFalse(design["caller_declared_source_metadata_only"])
        self.assertEqual(design["official_collection_inputs"], inputs)
        valid, errors = verify_experiment_c_external_rescore(
            output,
            source_run_dir=self.run_dir,
        )
        self.assertTrue(valid, errors)

        with self.assertRaisesRegex(
            ValueError,
            "requires --external-collection-dir",
        ):
            import_experiment_c_external_rescore(
                run_dir=self.run_dir,
                judgments_path=self.judgments_path,
                output_dir=self.root / "review-no-provenance-mode",
                allow_reviewed_generic_decoders=False,
            )

    def test_openrouter_collection_retains_gateway_provenance_boundary(
        self,
    ) -> None:
        collection = self.root / "mock-openrouter-collection"
        collection.mkdir()
        output = self.root / "review-openrouter-provenance"
        inputs = {
            "decoder_judgments": {
                "filename": "judgments.jsonl",
                "sha256": "b" * 64,
                "bytes": self.judgments_path.stat().st_size,
                "record_count": len(self.judgments),
            }
        }
        summary = {
            "provenance_mode": "selected_openrouter_gateway_collection",
            "collection_status": "complete",
            "gateway": "openrouter",
            "shared_gateway": True,
            "first_party_origin_claimed": False,
            "statistical_independence_claimed": False,
        }
        with patch.object(
            review_module,
            "validate_selected_external_decoder_collection",
            return_value=(self.judgments, inputs, summary),
        ) as validate:
            result = import_experiment_c_external_rescore(
                run_dir=self.run_dir,
                judgments_path=self.judgments_path,
                output_dir=output,
                external_collection_dir=collection,
                allow_reviewed_generic_decoders=False,
            )
        self.assertEqual(validate.call_count, 2)
        self.assertFalse(result["provider_provenance_validated"])
        self.assertTrue(result["gateway_provenance_validated"])
        review = json.loads(
            (output / "review.json").read_text(encoding="utf-8")
        )
        design = review["validation"]["source_design"]
        self.assertEqual(
            design["provenance_mode"],
            "selected_openrouter_gateway_collection",
        )
        self.assertFalse(design["provider_provenance_validated"])
        self.assertTrue(design["gateway_provenance_validated"])
        self.assertFalse(design["first_party_origin_claimed"])
        self.assertTrue(design["shared_gateway"])
        self.assertFalse(design["distinct_transport_origins"])
        boundary = review["interpretation_boundary"]
        self.assertIn("audit-validated through OpenRouter", boundary)
        self.assertIn("shares one gateway", boundary)
        self.assertIn("no direct first-party provider origin", boundary)
        valid, errors = verify_experiment_c_external_rescore(
            output,
            source_run_dir=self.run_dir,
        )
        self.assertTrue(valid, errors)

    def test_cli_requires_an_explicit_decoder_provenance_mode(self) -> None:
        parser = build_parser()
        base = [
            "experiment-c-decoder",
            "import",
            str(self.run_dir),
            str(self.judgments_path),
            str(self.root / "unused-cli-review"),
        ]
        with self.assertRaises(SystemExit) as missing:
            with redirect_stderr(StringIO()):
                parser.parse_args(base)
        self.assertEqual(missing.exception.code, 2)

        generic = parser.parse_args(
            [*base, "--allow-reviewed-generic-decoders"]
        )
        self.assertTrue(generic.allow_reviewed_generic_decoders)
        self.assertIsNone(generic.external_collection_dir)
        self.assertIsNone(generic.external_collection_provenance_mode)
        official = parser.parse_args(
            [*base, "--external-collection-dir", str(self.root)]
        )
        self.assertFalse(official.allow_reviewed_generic_decoders)
        self.assertEqual(official.external_collection_dir, self.root)
        self.assertEqual(
            official.external_collection_provenance_mode,
            DIRECT_FIRST_PARTY_COLLECTION_PROVENANCE,
        )
        openrouter = parser.parse_args(
            [*base, "--openrouter-collection-dir", str(self.root)]
        )
        self.assertFalse(openrouter.allow_reviewed_generic_decoders)
        self.assertEqual(openrouter.external_collection_dir, self.root)
        self.assertEqual(
            openrouter.external_collection_provenance_mode,
            OPENROUTER_COLLECTION_PROVENANCE,
        )

    def test_import_fails_closed_on_incomplete_or_third_source(self) -> None:
        incomplete = self.root / "incomplete.jsonl"
        self._write_judgments(incomplete, self.judgments[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete coverage"):
            import_experiment_c_external_rescore(
                run_dir=self.run_dir,
                judgments_path=incomplete,
                output_dir=self.root / "review-incomplete",
            )
        third = tuple(
            ExternalDecoderJudgment(
                request_id=request.request_id,
                request_sha256=request.request_sha256,
                decoder_instance_id="decoder-c",
                decoder_family_id="family-c",
                judgment_origin="external_model",
                source_descriptor="source-c",
                blind_to_system_identity=True,
                blind_to_latent_truth=True,
                probabilities=(
                    (0.25, 0.25, 0.25, 0.25),
                    (0.25, 0.25, 0.25, 0.25),
                    (0.25, 0.25, 0.25, 0.25),
                ),
            )
            for request in self.requests
        )
        third_path = self.root / "third.jsonl"
        self._write_judgments(
            third_path,
            self.judgments + third,
        )
        with self.assertRaisesRegex(ValueError, "exactly two judgments"):
            import_experiment_c_external_rescore(
                run_dir=self.run_dir,
                judgments_path=third_path,
                output_dir=self.root / "review-third",
            )

    def test_tampered_source_or_review_fails_closed(self) -> None:
        copied_parent = self.root / "copied"
        copied_parent.mkdir()
        copied = copied_parent / self.run_dir.name
        shutil.copytree(self.run_dir, copied)
        request_path = copied / PACKET_REQUESTS
        request_path.write_text(
            request_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "verification failed"):
            import_experiment_c_external_rescore(
                run_dir=copied,
                judgments_path=self.judgments_path,
                output_dir=self.root / "review-tampered-source",
            )

        output = self.root / "review-tampered-output"
        import_experiment_c_external_rescore(
            run_dir=self.run_dir,
            judgments_path=self.judgments_path,
            output_dir=output,
        )
        ranking = output / "metrics/experiment-c-rankings.json"
        ranking.write_text(
            ranking.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        valid, errors = verify_experiment_c_external_rescore(output)
        self.assertFalse(valid)
        self.assertTrue(
            any("checksum mismatch" in error for error in errors)
        )

    def test_publication_is_staged_and_write_failure_leaves_no_partial(self) -> None:
        atomic_output = self.root / "review-atomic-visibility"
        original_write = review_module._write_json
        output_seen_during_write: list[bool] = []

        def observe_write(path: Path, value: object) -> None:
            output_seen_during_write.append(atomic_output.exists())
            original_write(path, value)

        with patch.object(
            review_module,
            "_write_json",
            side_effect=observe_write,
        ):
            import_experiment_c_external_rescore(
                run_dir=self.run_dir,
                judgments_path=self.judgments_path,
                output_dir=atomic_output,
            )
        self.assertTrue(output_seen_during_write)
        self.assertFalse(any(output_seen_during_write))
        self.assertTrue(atomic_output.is_dir())

        failed_output = self.root / "review-injected-write-failure"
        writes = 0

        def fail_after_write(path: Path, value: object) -> None:
            nonlocal writes
            original_write(path, value)
            writes += 1
            if writes == 2:
                raise OSError("injected staged write failure")

        with patch.object(
            review_module,
            "_write_json",
            side_effect=fail_after_write,
        ):
            with self.assertRaisesRegex(
                OSError,
                "injected staged write failure",
            ):
                import_experiment_c_external_rescore(
                    run_dir=self.run_dir,
                    judgments_path=self.judgments_path,
                    output_dir=failed_output,
                )
        self._assert_no_partial_output(failed_output)

        raced_output = self.root / "review-destination-race"
        destination_created = False

        def create_destination(path: Path, value: object) -> None:
            nonlocal destination_created
            if not destination_created:
                raced_output.mkdir()
                (raced_output / "owner-marker.txt").write_text(
                    "unrelated owner\n",
                    encoding="utf-8",
                )
                destination_created = True
            original_write(path, value)

        with patch.object(
            review_module,
            "_write_json",
            side_effect=create_destination,
        ):
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                import_experiment_c_external_rescore(
                    run_dir=self.run_dir,
                    judgments_path=self.judgments_path,
                    output_dir=raced_output,
                )
        self.assertEqual(
            (raced_output / "owner-marker.txt").read_text(encoding="utf-8"),
            "unrelated owner\n",
        )
        self.assertFalse(
            (
                raced_output.parent
                / f".{raced_output.name}.external-rescore.lock"
            ).exists()
        )
        self.assertEqual(
            list(
                raced_output.parent.glob(
                    f".{raced_output.name}.*.staging"
                )
            ),
            [],
        )

    def test_final_source_and_judgment_reverification_fails_atomically(
        self,
    ) -> None:
        original_write = review_module._write_json
        judgment_output = self.root / "review-judgment-race"
        judgment_material = self.judgments_path.read_bytes()
        mutated_judgment = False

        def mutate_judgment(path: Path, value: object) -> None:
            nonlocal mutated_judgment
            if not mutated_judgment:
                self.judgments_path.write_bytes(judgment_material + b"\n")
                mutated_judgment = True
            original_write(path, value)

        try:
            with patch.object(
                review_module,
                "_write_json",
                side_effect=mutate_judgment,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "judgments changed while the import was running",
                ):
                    import_experiment_c_external_rescore(
                        run_dir=self.run_dir,
                        judgments_path=self.judgments_path,
                        output_dir=judgment_output,
                    )
        finally:
            self.judgments_path.write_bytes(judgment_material)
        self._assert_no_partial_output(judgment_output)

        copied_parent = self.root / "source-race-copy"
        copied_parent.mkdir()
        copied = copied_parent / self.run_dir.name
        shutil.copytree(self.run_dir, copied)
        source_output = self.root / "review-source-race"
        mutated_source = False

        def mutate_source(path: Path, value: object) -> None:
            nonlocal mutated_source
            if not mutated_source:
                source_requests = copied / PACKET_REQUESTS
                source_requests.write_text(
                    source_requests.read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )
                mutated_source = True
            original_write(path, value)

        with patch.object(
            review_module,
            "_write_json",
            side_effect=mutate_source,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "source run changed while the import was running",
            ):
                import_experiment_c_external_rescore(
                    run_dir=copied,
                    judgments_path=self.judgments_path,
                    output_dir=source_output,
                )
        self._assert_no_partial_output(source_output)

    def test_staged_self_verification_lock_and_symlink_guards(self) -> None:
        verification_output = self.root / "review-staged-verification-failure"
        with patch.object(
            review_module,
            "verify_experiment_c_external_rescore",
            return_value=(False, ("injected staged verification failure",)),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "staged external rescore failed verification",
            ):
                import_experiment_c_external_rescore(
                    run_dir=self.run_dir,
                    judgments_path=self.judgments_path,
                    output_dir=verification_output,
                )
        self._assert_no_partial_output(verification_output)

        locked_output = self.root / "review-locked"
        lock = (
            locked_output.parent
            / f".{locked_output.name}.external-rescore.lock"
        )
        lock.write_text("held\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(FileExistsError, "is locked"):
                import_experiment_c_external_rescore(
                    run_dir=self.run_dir,
                    judgments_path=self.judgments_path,
                    output_dir=locked_output,
                )
            self.assertFalse(locked_output.exists())
        finally:
            lock.unlink()

        judgment_link = self.root / "judgments-link.jsonl"
        judgment_link.symlink_to(self.judgments_path)
        with self.assertRaisesRegex(ValueError, "safe regular JSONL"):
            import_experiment_c_external_rescore(
                run_dir=self.run_dir,
                judgments_path=judgment_link,
                output_dir=self.root / "review-judgment-link",
            )
        run_link = self.root / "run-link"
        run_link.symlink_to(self.run_dir, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "source run cannot be a symlink"):
            import_experiment_c_external_rescore(
                run_dir=run_link,
                judgments_path=self.judgments_path,
                output_dir=self.root / "review-run-link",
            )
        dangling_target = self.root / "not-created-output-target"
        output_link = self.root / "review-output-link"
        output_link.symlink_to(dangling_target, target_is_directory=True)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            import_experiment_c_external_rescore(
                run_dir=self.run_dir,
                judgments_path=self.judgments_path,
                output_dir=output_link,
            )

    def test_new_public_schemas_cover_runtime_rows(self) -> None:
        codebook = self._jsonl(self.run_dir / PACKET_CODEBOOK)[0]
        schema = SCHEMAS["experiment-c-decoder-codebook"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(codebook), set(schema["required"]))
        self.assertEqual(set(codebook), set(schema["properties"]))

        output = self.root / "review-schema"
        import_experiment_c_external_rescore(
            run_dir=self.run_dir,
            judgments_path=self.judgments_path,
            output_dir=output,
        )
        score = self._jsonl(
            output / "metrics/external-decoder-scores.jsonl"
        )[0]
        score_schema = SCHEMAS["experiment-c-external-score"]
        self.assertFalse(score_schema["additionalProperties"])
        self.assertEqual(set(score), set(score_schema["required"]))
        self.assertEqual(set(score), set(score_schema["properties"]))


if __name__ == "__main__":
    unittest.main()
