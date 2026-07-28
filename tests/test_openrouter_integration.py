from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import json
import unittest

from cape_loop.cli import build_parser, main as cli_main
from cape_loop.config import (
    AppConfig,
    ConfigError,
    ExperimentSection,
    InferenceSection,
    LLMSection,
    RunSection,
    load_config,
)
from cape_loop.decoder_study import ExternalDecoderRequest
from cape_loop.llm_exchange import (
    ATTRIBUTES,
    VALUES,
    LLMRequest,
    LLMResponse,
    write_requests,
)
from cape_loop.openrouter_provider import (
    OPENROUTER_EXAMPLE_MODEL,
    OpenRouterChatProvider,
    PreparedOpenRouterRequest,
    ResumableOpenRouterCompletionProvider,
)
from cape_loop.runner import (
    _live_completion_provider,
    _llm_input_manifest,
    run_experiment,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "live"
    / "experiment_a_openrouter.toml"
)


def profile_request() -> LLMRequest:
    return LLMRequest.build(
        request_id="openrouter-plan-one",
        updater_id="llm_full_context",
        view="full_context",
        prior={},
        observation={"selected_option": "a"},
        context={"options": ["a", "b"]},
    )


class OpenRouterConfigurationTests(unittest.TestCase):
    def test_mode_sensitive_defaults_and_checked_config(self) -> None:
        config = AppConfig.parse(
            {
                "schema_version": 1,
                "llm": {
                    "mode": "openrouter",
                    "model": OPENROUTER_EXAMPLE_MODEL,
                },
            }
        )
        self.assertEqual(config.llm.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(
            config.llm.base_url,
            "https://openrouter.ai/api",
        )
        self.assertEqual(config.llm.max_retries, 2)

        checked = load_config(OPENROUTER_CONFIG)
        self.assertEqual(checked.llm.mode, "openrouter")
        self.assertEqual(
            checked.llm.model,
            "google/gemini-3.6-flash",
        )
        self.assertEqual(
            checked.llm.openrouter_upstream_provider,
            "google-vertex/global",
        )

    def test_noncanonical_or_unsafe_routes_fail_closed(self) -> None:
        invalid_sections = (
            {"mode": "openrouter"},
            {
                "mode": "openrouter",
                "model": "~google/gemini-flash-latest",
            },
            {
                "mode": "openrouter",
                "model": "google/gemini-3.6-flash:free",
            },
            {
                "mode": "openrouter",
                "model": OPENROUTER_EXAMPLE_MODEL,
                "base_url": "https://api.openai.com",
            },
            {
                "mode": "openrouter",
                "model": OPENROUTER_EXAMPLE_MODEL,
                "base_url": "https://proxy.example.test",
                "allow_custom_base_url": True,
            },
        )
        for section in invalid_sections:
            with self.subTest(section=section), self.assertRaises(
                ConfigError
            ):
                AppConfig.parse(
                    {
                        "schema_version": 1,
                        "llm": section,
                    }
                )
        with self.assertRaisesRegex(ConfigError, "only in OpenRouter"):
            AppConfig.parse(
                {
                    "schema_version": 1,
                    "llm": {
                        "mode": "openai",
                        "reasoning_effort": "minimal",
                    },
                }
            )

    def test_input_manifest_is_gateway_explicit(self) -> None:
        config = load_config(OPENROUTER_CONFIG)
        manifest = _llm_input_manifest(config)
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest["provider"], "openrouter")
        self.assertEqual(
            manifest["endpoint"],
            "https://openrouter.ai/api/v1/chat/completions",
        )
        self.assertFalse(manifest["response_cache_enabled"])
        self.assertTrue(manifest["router_metadata_requested"])
        self.assertFalse(manifest["router_transforms_accepted"])
        self.assertFalse(manifest["first_party_origin_claimed"])
        self.assertEqual(
            manifest["provider_preferences"]["only"],
            ["google-vertex/global"],
        )

    def test_runner_requires_explicit_live_authorization(self) -> None:
        with TemporaryDirectory() as directory:
            config = AppConfig(
                run=RunSection(
                    name="openrouter-no-live",
                    output_root=directory,
                    deterministic=False,
                ),
                experiment=ExperimentSection(
                    kind="provenance_audit",
                    domains=("travel",),
                    mechanisms=("balanced", "restricted"),
                    response_modes=("naturally_sampled",),
                    policies=("balanced",),
                    updaters=(
                        "fitted_action_aware",
                        "llm_full_context",
                    ),
                    users=1,
                    trajectories_per_cell=1,
                    turns=1,
                    bootstrap_replicates=0,
                ),
                inference=InferenceSection(
                    training_interactions=8,
                    fit_steps=2,
                    learning_rate=0.02,
                    l2=0.0,
                    calibration="none",
                ),
                llm=LLMSection(
                    mode="openrouter",
                    model=OPENROUTER_EXAMPLE_MODEL,
                    api_key_env="OPENROUTER_API_KEY",
                    base_url="https://openrouter.ai/api",
                ),
            )
            with self.assertRaisesRegex(ValueError, "--execute-live"):
                run_experiment(config)
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_runner_dispatches_to_separate_openrouter_journal(self) -> None:
        config = load_config(OPENROUTER_CONFIG)
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "run-id"
            adapter = _live_completion_provider(
                config,
                destination=destination,
                execute_live=True,
            )
            self.assertIsInstance(
                adapter,
                ResumableOpenRouterCompletionProvider,
            )
            assert adapter is not None
            self.assertIn("openrouter", adapter.responses_path.parts)
            self.assertEqual(
                adapter.provider.config.api_key_env,
                "OPENROUTER_API_KEY",
            )
            self.assertEqual(
                adapter.provider.config.upstream_provider,
                "google-vertex/global",
            )


class OpenRouterCLIIntegrationTests(unittest.TestCase):
    def test_parser_exposes_safe_openrouter_defaults(self) -> None:
        args = build_parser().parse_args(
            ["llm", "plan-openrouter", "requests.jsonl"]
        )
        self.assertEqual(args.model, OPENROUTER_EXAMPLE_MODEL)
        self.assertEqual(args.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(args.base_url, "https://openrouter.ai/api")
        self.assertFalse(args.allow_fallbacks)
        self.assertFalse(args.allow_unsupported_parameters)
        self.assertEqual(args.data_collection, "deny")

    def test_plan_is_keyless_and_has_auditable_route(self) -> None:
        with TemporaryDirectory() as directory:
            requests = Path(directory) / "requests.jsonl"
            write_requests(requests, (profile_request(),))
            stdout = io.StringIO()
            with (
                patch.dict(
                    "os.environ",
                    {"OPENROUTER_API_KEY": "must-not-be-read"},
                    clear=False,
                ),
                patch(
                    "cape_loop.openrouter_provider.build_opener",
                    side_effect=AssertionError("plan attempted network"),
                ),
                redirect_stdout(stdout),
            ):
                status = cli_main(
                    [
                        "llm",
                        "plan-openrouter",
                        str(requests),
                        "--upstream-provider",
                        "google-ai-studio",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["provider"], "openrouter")
            self.assertFalse(payload["credential_read"])
            self.assertEqual(
                payload["provider_preferences"]["only"],
                ["google-ai-studio"],
            )
            self.assertFalse(payload["response_cache_enabled"])
            self.assertFalse(payload["first_party_origin_claimed"])

    def test_live_command_requires_explicit_flag(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests = root / "requests.jsonl"
            write_requests(requests, (profile_request(),))
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                cli_main(
                    [
                        "llm",
                        "execute-openrouter",
                        str(requests),
                        str(root / "responses.jsonl"),
                        str(root / "audit.jsonl"),
                    ]
                )
            self.assertIn("--execute-live", stderr.getvalue())

    def test_decoder_plan_matches_live_request_identity_body_and_bound(
        self,
    ) -> None:
        request = ExternalDecoderRequest.build(
            request_id="openrouter-decoder",
            pseudonymous_state_id="state-openrouter",
            representation_id="blinded-native-content-v1",
            evaluation_split="development",
            payload={
                "representation_version": "blinded-native-content-v1",
                "episodes": [],
                "semantic_claims": [],
                "persona_text": "",
            },
        )
        dispatched_requests: list[LLMRequest] = []
        dispatched_preparations: list[PreparedOpenRouterRequest] = []

        class CapturingAdapter:
            def __init__(
                self,
                provider: OpenRouterChatProvider,
                *,
                responses_path: Path,
                audit_path: Path,
                **_: object,
            ) -> None:
                self.provider = provider
                self.responses_path = responses_path
                self.audit_path = audit_path
                self.attempts_path = audit_path.with_name(
                    "provider-audit-transport-attempts.jsonl"
                )
                responses_path.parent.mkdir(parents=True, exist_ok=True)
                for path in (
                    self.responses_path,
                    self.audit_path,
                    self.attempts_path,
                ):
                    path.write_text("", encoding="utf-8")

            def require_static_corpus_capacity(
                self,
                requests: object,
            ) -> dict[str, int]:
                del requests
                return {}

            def complete(self, provider_request: LLMRequest) -> LLMResponse:
                dispatched_requests.append(provider_request)
                dispatched_preparations.append(
                    self.provider.prepare(provider_request)
                )
                return LLMResponse(
                    request_id=provider_request.request_id,
                    prompt_sha256=provider_request.prompt_sha256,
                    model_id=self.provider.config.model,
                    beliefs={
                        attribute: {value: 0.25 for value in VALUES}
                        for attribute in ATTRIBUTES
                    },
                )

            @property
            def used_audit_records(self) -> tuple[object, ...]:
                return ()

            @property
            def used_attempt_records(self) -> tuple[object, ...]:
                return ()

            def to_manifest(self) -> dict[str, object]:
                return {
                    "schema_version": 1,
                    "model": self.provider.config.model,
                }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "decoder.jsonl"
            path.write_text(
                json.dumps(request.to_dict(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = cli_main(
                    [
                        "decoder-study",
                        "plan-openrouter",
                        str(path),
                        "--model",
                        OPENROUTER_EXAMPLE_MODEL,
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(stdout.getvalue())

            execution_stdout = io.StringIO()
            with (
                patch(
                    "cape_loop.cli.ResumableOpenRouterCompletionProvider",
                    CapturingAdapter,
                ),
                redirect_stdout(execution_stdout),
            ):
                execution_status = cli_main(
                    [
                        "decoder-study",
                        "execute-openrouter",
                        str(path),
                        str(root / "output"),
                        "--model",
                        OPENROUTER_EXAMPLE_MODEL,
                        "--execute-live",
                    ]
                )

            self.assertEqual(execution_status, 0)
            self.assertFalse(
                payload["strict_first_party_gate4_eligible"]
            )
            self.assertFalse(payload["statistical_independence_claimed"])
            self.assertFalse(payload["first_party_origin_claimed"])
            self.assertEqual(len(dispatched_requests), 1)
            self.assertEqual(len(dispatched_preparations), 1)

            model = OPENROUTER_EXAMPLE_MODEL
            model_digest = sha256(model.encode("utf-8")).hexdigest()[:12]
            instance_id = f"openrouter-{model_digest}"
            provider_request = dispatched_requests[0]
            prepared = dispatched_preparations[0]
            source = payload["sources"][0]
            self.assertEqual(source["decoder_instance_id"], instance_id)
            self.assertEqual(
                provider_request.request_id,
                (
                    f"external-decoder:{instance_id}:"
                    f"{request.request_sha256}"
                ),
            )
            self.assertEqual(
                source["request_body_sha256"],
                [
                    {
                        "request_id": request.request_id,
                        "provider_request_id": provider_request.request_id,
                        "sha256": prepared.body_sha256,
                        "estimated_max_tokens": (
                            prepared.estimated_max_tokens
                        ),
                    }
                ],
            )
            self.assertEqual(
                source["conservative_max_tokens"],
                prepared.estimated_max_tokens,
            )


if __name__ == "__main__":
    unittest.main()
