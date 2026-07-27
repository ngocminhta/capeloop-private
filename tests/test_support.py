from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cape_loop.artifacts import RunArtifacts, source_tree_digest, verify_run
from cape_loop.calibration import CalibrationExample, fit_temperature
from cape_loop.cli import main as cli_main
from cape_loop.config import (
    AppConfig,
    ConfigError,
    ExperimentSection,
    LLMSection,
    load_config,
)
from cape_loop.beliefs import PreferenceBelief
from cape_loop.domains import TRAVEL
from cape_loop.elicitation import build_matched_anchor_set
from cape_loop.human_study import (
    StudyItem,
    blind_and_order_items,
    build_assignment_codebook,
    validate_rating_record,
)
from cape_loop.llm_exchange import LLMRequest, LLMResponse, ReplayProvider
from cape_loop.policies import SoftProfileConditionedPolicy
from cape_loop.schema_export import export_schemas
from cape_loop.runner import _llm_input_manifest
from cape_loop.splits import (
    assert_terminal_templates_held_out,
    build_split_manifest,
)
from cape_loop.updaters import (
    LLMReplayUpdater,
    UpdateViewKind,
    make_update_view,
)
from cape_loop.schemas import Observation, PolicyProvenance
from cape_loop.verbalization import (
    allowed_verbalizations,
    validate_surface_response,
    verbalize_choice,
)


class ConfigTests(unittest.TestCase):
    def test_load_minimal_config_and_reject_unknown_key(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.toml"
            path.write_text(
                'schema_version = 1\n[run]\nname = "test"\nseed = 4\n',
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.run.name, "test")
            path.write_text(
                'schema_version = 1\n[run]\nname = "test"\ntyop = true\n',
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_config_rejects_duplicate_nonfinite_and_boolean_cells(self) -> None:
        with self.assertRaises(ConfigError):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "experiment": {"domains": ["travel", "travel"]},
                }
            )
        with self.assertRaises(ConfigError):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "thresholds": {
                        "direction_tolerance": float("nan"),
                    },
                }
            )
        with self.assertRaises(ConfigError):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "experiment": {"users": True},
                }
            )
        with self.assertRaises(ConfigError):
            AppConfig(
                experiment=ExperimentSection(users=True),
            ).validated()

    def test_sensitivity_rejects_point_specific_llm_calibration(self) -> None:
        config = AppConfig(
            experiment=ExperimentSection(
                kind="sensitivity",
                mechanisms=("ranking", "default", "suggestion"),
                response_modes=("naturally_sampled",),
                policies=("balanced", "soft_profile_conditioned"),
                updaters=("llm_full_context",),
                bootstrap_replicates=0,
            ),
            llm=LLMSection(responses_file="responses.jsonl"),
        )
        with self.assertRaisesRegex(
            ConfigError,
            "llm.calibration = 'none'",
        ):
            config.validate_experiment_contract()

    def test_sensitivity_rejects_inactive_generic_turn_count(self) -> None:
        config = AppConfig(
            experiment=ExperimentSection(
                kind="sensitivity",
                mechanisms=("ranking", "default", "suggestion"),
                response_modes=("naturally_sampled",),
                policies=("balanced", "soft_profile_conditioned"),
                turns=8,
                bootstrap_replicates=0,
            ),
        )
        with self.assertRaises(ConfigError):
            config.validate_experiment_contract()

    def test_provenance_audit_accepts_confirmatory_bootstrap_replicates(
        self,
    ) -> None:
        config = AppConfig(
            experiment=ExperimentSection(
                kind="provenance_audit",
                mechanisms=("balanced", "restricted"),
                response_modes=("controlled", "naturally_sampled"),
                policies=("balanced",),
                trajectories_per_cell=1,
                turns=1,
                bootstrap_replicates=2_000,
            ),
        )
        config.validate_experiment_contract()

    def test_prior_strengths_are_validated_and_scoped_to_experiment_a(
        self,
    ) -> None:
        parsed = AppConfig.parse(
            {
                "schema_version": 1,
                "experiment": {
                    "prior_strengths": [0.0, 0.35, 0.7],
                },
            }
        )
        self.assertEqual(
            parsed.experiment.prior_strengths,
            (0.0, 0.35, 0.7),
        )
        with self.assertRaises(ConfigError):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "experiment": {
                        "prior_strengths": [0.0, 1.0],
                    },
                }
            )
        with self.assertRaises(ConfigError):
            AppConfig(
                experiment=ExperimentSection(
                    kind="closed_loop",
                    mechanisms=("ranking", "default", "suggestion"),
                    response_modes=("naturally_sampled",),
                    policies=("balanced",),
                    prior_strengths=(0.0, 0.5),
                )
            ).validate_experiment_contract()


class CalibrationTests(unittest.TestCase):
    def test_calibration_rejects_test_split(self) -> None:
        examples = [
            CalibrationExample((0.8, 0.2), 0, "development"),
            CalibrationExample((0.2, 0.8), 1, "development"),
        ]
        result = fit_temperature(examples)
        self.assertGreater(result.temperature, 0)
        self.assertAlmostEqual(sum(result.apply((0.25, 0.75))), 1.0)
        with self.assertRaises(ValueError):
            fit_temperature(examples, allowed_splits=("test",))


class ExchangeTests(unittest.TestCase):
    def test_view_boundaries_and_strict_response_validation(self) -> None:
        request = LLMRequest.build(
            request_id="r1",
            updater_id="full-context",
            view="full_context",
            prior={"attribute_1": {"-2": 0.25, "-1": 0.25, "+1": 0.25, "+2": 0.25}},
            observation={"selected_option": "anchor"},
            context={"options": ["anchor", "alternative"]},
            provenance={"secret": "must-not-leak"},
        )
        self.assertNotIn("provenance", request.payload)
        response = LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": "r1",
                "prompt_sha256": request.prompt_sha256,
                "model_id": "replay-model",
                "beliefs": {
                    "attribute_1": {
                        "-2": 0.1,
                        "-1": 0.2,
                        "+1": 0.3,
                        "+2": 0.4,
                    },
                    "attribute_2": {
                        "-2": 0.25,
                        "-1": 0.25,
                        "+1": 0.25,
                        "+2": 0.25,
                    },
                    "attribute_3": {
                        "-2": 0.25,
                        "-1": 0.25,
                        "+1": 0.25,
                        "+2": 0.25,
                    },
                },
            }
        )
        self.assertEqual(ReplayProvider([response]).complete(request), response)
        malformed = json.loads(json.dumps(response.to_dict()))
        malformed["beliefs"]["attribute_1"]["+2"] = 0.6
        with self.assertRaises(ValueError):
            LLMResponse.parse(malformed)
        for invalid_probability in (True, float("nan")):
            malformed = json.loads(json.dumps(response.to_dict()))
            malformed["beliefs"]["attribute_1"] = {
                "-2": invalid_probability,
                "-1": 0.0,
                "+1": 0.0,
                "+2": 1.0,
            }
            with self.assertRaises(ValueError):
                LLMResponse.parse(malformed)
        malformed = json.loads(json.dumps(response.to_dict()))
        malformed["raw_response_sha256"] = "not-a-digest"
        with self.assertRaises(ValueError):
            LLMResponse.parse(malformed)

    def test_llm_replay_updater_binds_prompt_and_updates_profile(self) -> None:
        matched = build_matched_anchor_set(TRAVEL, scenario_id="llm-replay")
        context = matched.context("default")
        view = make_update_view(
            UpdateViewKind.FULL_CONTEXT,
            context,
            matched.observation(),
            PolicyProvenance("test-policy", "v1"),
            event_id="event-1",
        )
        prior = PreferenceBelief.uniform()
        request_builder = LLMReplayUpdater(
            "llm_full_context",
            UpdateViewKind.FULL_CONTEXT,
            ReplayProvider(()),
        )
        state = request_builder.initial_state(prior)
        request = request_builder.build_request(state, view)
        beliefs = {
            f"attribute_{attribute}": {
                "-2": 0.1,
                "-1": 0.1,
                "+1": 0.3,
                "+2": 0.5,
            }
            for attribute in range(1, 4)
        }
        response = LLMResponse.parse(
            {
                "schema_version": 1,
                "request_id": request.request_id,
                "prompt_sha256": request.prompt_sha256,
                "model_id": "fixture-model-v1",
                "beliefs": beliefs,
            }
        )
        updater = LLMReplayUpdater(
            "llm_full_context",
            UpdateViewKind.FULL_CONTEXT,
            ReplayProvider((response,)),
        )
        result = updater.update(updater.initial_state(prior), view)
        self.assertGreater(result.state.belief.expected_theta()[0], 0)
        self.assertEqual(updater.requests, (request,))
        mismatched = LLMResponse.parse(
            {
                **response.to_dict(),
                "prompt_sha256": "0" * 64,
            }
        )
        with self.assertRaises(ValueError):
            ReplayProvider((mismatched,)).complete(request)

    def test_llm_prompt_projection_excludes_audit_and_policy_labels(self) -> None:
        hidden = "user-7:incorrect:secret-crn"
        action = SoftProfileConditionedPolicy().action(
            TRAVEL,
            PreferenceBelief.uniform(),
            turn=0,
            master_seed=9,
            trajectory_id=hidden,
        )
        observation = Observation(
            selected_option_id=action.context.ranking[0],
            surface_response="I choose this option.",
            choice_noise_key=f"closed-loop:{hidden}",
        )
        for view_kind, updater_id in (
            (UpdateViewKind.RESPONSE_ONLY, "llm_response_only"),
            (UpdateViewKind.FULL_CONTEXT, "llm_full_context"),
        ):
            view = make_update_view(
                view_kind,
                action.context,
                observation,
                action.provenance,
                event_id=f"event:{hidden}:soft_profile_conditioned",
            )
            updater = LLMReplayUpdater(
                updater_id,
                view_kind,
                ReplayProvider(()),
            )
            request = updater.build_request(
                updater.initial_state(PreferenceBelief.uniform()),
                view,
            )
            serialized = json.dumps(
                {
                    "request_id": request.request_id,
                    "payload": request.payload,
                },
                sort_keys=True,
            )
            for forbidden in (
                "choice_noise_key",
                "context_id",
                "scenario_id",
                "turn_id",
                "incorrect",
                "secret-crn",
                "user-7",
                "soft_profile_conditioned",
            ):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(
                request.request_id,
                f"{updater_id}:{request.prompt_sha256}",
            )


class ReproducibilitySupportTests(unittest.TestCase):
    def test_external_llm_replay_corpus_is_fingerprinted(self) -> None:
        with TemporaryDirectory() as directory:
            response_path = Path(directory) / "responses.jsonl"
            response = {
                "schema_version": 1,
                "request_id": "request-1",
                "prompt_sha256": "0" * 64,
                "model_id": "fixture-model-v1",
                "beliefs": {
                    f"attribute_{attribute}": {
                        "-2": 0.25,
                        "-1": 0.25,
                        "+1": 0.25,
                        "+2": 0.25,
                    }
                    for attribute in range(1, 4)
                },
            }
            response_path.write_text(
                json.dumps(response) + "\n",
                encoding="utf-8",
            )
            config = AppConfig(
                experiment=ExperimentSection(
                    updaters=("llm_full_context",),
                ),
                llm=LLMSection(responses_file=str(response_path)),
            )
            first = _llm_input_manifest(config)
            self.assertIsNotNone(first)
            self.assertEqual(first["response_count"], 1)
            response["model_id"] = "fixture-model-v2"
            response_path.write_text(
                json.dumps(response) + "\n",
                encoding="utf-8",
            )
            second = _llm_input_manifest(config)
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_split_manifest_is_group_disjoint_and_stable(self) -> None:
        first = build_split_manifest(seed=9)
        second = build_split_manifest(seed=9)
        first.assert_disjoint()
        self.assertEqual(first.to_dict(), second.to_dict())
        assert_terminal_templates_held_out(first, ["terminal-v1"])
        self.assertEqual(
            first.dialogue_templates["direct-probe-v1"],
            "test",
        )

    def test_artifact_checksum_detects_mutation(self) -> None:
        config = AppConfig()
        with TemporaryDirectory() as directory:
            run = RunArtifacts.create(config, root=directory)
            run.write_jsonl("events/events.jsonl", [{"event": 1}])
            run.finalize({"status": "smoke"})
            ok, errors = verify_run(run.path)
            self.assertTrue(ok, errors)
            (run.path / "metrics" / "summary.json").write_text(
                "{}\n", encoding="utf-8"
            )
            ok, errors = verify_run(run.path)
            self.assertFalse(ok)
            self.assertTrue(any("checksum mismatch" in error for error in errors))

    def test_artifact_checksum_rejects_escape_and_unlisted_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_path = root / "run"
            run_path.mkdir()
            outside = root / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            (run_path / "SHA256SUMS").write_text(
                f"{'0' * 64}  ../outside.json\n",
                encoding="utf-8",
            )
            ok, errors = verify_run(run_path)
            self.assertFalse(ok)
            self.assertTrue(
                any("unsafe checksum path" in error for error in errors)
            )

            run = RunArtifacts.create(AppConfig(), root=root / "artifacts")
            run.finalize({"status": "smoke"})
            (run.path / "unlisted.txt").write_text("unlisted\n", encoding="utf-8")
            ok, errors = verify_run(run.path)
            self.assertFalse(ok)
            self.assertIn("unlisted artifact: unlisted.txt", errors)

    def test_artifact_checksum_rejects_alias_and_duplicate_paths(self) -> None:
        with TemporaryDirectory() as directory:
            run = RunArtifacts.create(AppConfig(), root=directory)
            run.finalize({"status": "smoke"})
            checksum_path = run.path / "SHA256SUMS"
            manifest_line = next(
                line
                for line in checksum_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.endswith("  manifest.json")
            )
            checksum_path.write_text(
                checksum_path.read_text(encoding="utf-8")
                + manifest_line.replace(
                    "  manifest.json",
                    "  events/../manifest.json",
                )
                + "\n",
                encoding="utf-8",
            )
            ok, errors = verify_run(run.path)
            self.assertFalse(ok)
            self.assertTrue(
                any("unsafe checksum path" in error for error in errors)
            )

            run.write_checksums()
            checksum_path.write_text(
                checksum_path.read_text(encoding="utf-8")
                + manifest_line
                + "\n",
                encoding="utf-8",
            )
            ok, errors = verify_run(run.path)
            self.assertFalse(ok)
            self.assertTrue(
                any("duplicate checksum path" in error for error in errors)
            )

    def test_artifact_verification_requires_completed_run_semantics(self) -> None:
        with TemporaryDirectory() as directory:
            run = RunArtifacts.create(AppConfig(), root=directory)
            run.write_checksums()
            ok, errors = verify_run(run.path)
            self.assertFalse(ok)
            self.assertIn("run manifest status is not complete", errors)
            self.assertIn("missing metrics/summary.json", errors)

    def test_source_tree_digest_is_stable(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        self.assertEqual(
            source_tree_digest(repository),
            source_tree_digest(repository),
        )

    def test_schema_export(self) -> None:
        with TemporaryDirectory() as directory:
            written = export_schemas(directory)
            self.assertGreaterEqual(len(written), 2)
            self.assertIn(
                "/2020-12/",
                json.loads(written[0].read_text())["$schema"],
            )
            repository_schemas = Path(__file__).resolve().parents[1] / "schemas"
            for generated in written:
                checked_in = repository_schemas / generated.name
                self.assertTrue(checked_in.is_file(), generated.name)
                self.assertEqual(generated.read_bytes(), checked_in.read_bytes())


class LanguageAndStudyTests(unittest.TestCase):
    def test_verbalizer_cannot_invent_general_preference(self) -> None:
        response = verbalize_choice("the first option")
        validate_surface_response(
            response,
            selected_label="the first option",
            allowed_responses=allowed_verbalizations("the first option"),
        )
        with self.assertRaises(ValueError):
            validate_surface_response(
                "I generally prefer the first option.",
                selected_label="the first option",
            )

    def test_study_packet_is_blinded_and_deterministic(self) -> None:
        items = [
            StudyItem(
                item_id="a",
                scenario_id="s",
                condition="balanced",
                vignette="The user selected the budget option.",
                preference_claim="the user generally prefers budget options",
            ),
            StudyItem(
                item_id="b",
                scenario_id="s",
                condition="default",
                vignette="The user kept the default budget option.",
                preference_claim="the user generally prefers budget options",
            ),
        ]
        first = blind_and_order_items(items, assignment_id="worker", seed=4)
        second = blind_and_order_items(items, assignment_id="worker", seed=4)
        self.assertEqual(first, second)
        self.assertNotIn("condition", first[0])
        self.assertNotIn("item_id", first[0])
        codebook = build_assignment_codebook(
            items,
            assignment_id="worker",
            seed=4,
        )
        self.assertEqual(set(codebook), {item["display_id"] for item in first})
        self.assertEqual(
            {entry["condition"] for entry in codebook.values()},
            {"balanced", "default"},
        )
        with self.assertRaises(ValueError):
            blind_and_order_items(items, assignment_id="", seed=4)
        with self.assertRaises(ValueError):
            validate_rating_record(
                {
                    "assignment_id": "",
                    "display_id": first[0]["display_id"],
                    "rating": 4,
                },
                valid_display_ids={item["display_id"] for item in first},
            )

    def test_human_study_cli_writes_complete_packet_without_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "packet"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "human-study",
                            "generate",
                            str(output),
                            "--assignment-id",
                            "test-assignment",
                            "--seed",
                            "19",
                        ]
                    ),
                    0,
                )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "README.md",
                    "human-rating.schema.json",
                    "order-manifest.json",
                    "packet-manifest.json",
                    "participant-items.jsonl",
                    "researcher-codebook.json",
                },
            )
            participant = json.loads(
                (output / "participant-items.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertNotIn("item_id", participant)
            self.assertNotIn("condition", participant)
            participant_rows = {
                row["display_id"]: row
                for row in (
                    json.loads(line)
                    for line in (
                        output / "participant-items.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                )
            }
            codebook = json.loads(
                (output / "researcher-codebook.json").read_text(
                    encoding="utf-8"
                )
            )
            volunteered_ids = {
                display_id
                for display_id, entry in codebook[
                    "items_by_display_id"
                ].items()
                if entry["condition"] == "volunteered"
            }
            self.assertEqual(len(volunteered_ids), 2)
            self.assertTrue(
                all(
                    'The user stated, "I generally prefer'
                    in participant_rows[display_id]["vignette"]
                    for display_id in volunteered_ids
                )
            )
            packet_manifest = json.loads(
                (output / "packet-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(packet_manifest["files"]),
                {
                    "README.md",
                    "human-rating.schema.json",
                    "order-manifest.json",
                    "participant-items.jsonl",
                    "researcher-codebook.json",
                },
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli_main(
                        [
                            "human-study",
                            "generate",
                            str(output),
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
