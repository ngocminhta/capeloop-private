"""Exported JSON Schemas for cross-language artifact consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


_SHA256 = {
    "type": "string",
    "pattern": "^[0-9a-f]{64}$",
}

_NONEMPTY_STRING = {"type": "string", "minLength": 1}

_SPLIT = {"enum": ["train", "development", "test"]}

_MECHANISM = {
    "enum": ["balanced", "restricted", "default", "suggested"]
}

_PROBABILITY_ROW = {
    "type": "object",
    "required": ["-2", "-1", "+1", "+2"],
    "additionalProperties": False,
    "properties": {
        value: {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        }
        for value in ("-2", "-1", "+1", "+2")
    },
}

_PROBABILITY_ROWS = {
    "type": "array",
    "items": _PROBABILITY_ROW,
    "minItems": 3,
    "maxItems": 3,
}

_LLM_RESPONSE_RECORD = {
    "type": "object",
    "required": [
        "schema_version",
        "request_id",
        "prompt_sha256",
        "model_id",
        "beliefs",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "request_id": _NONEMPTY_STRING,
        "prompt_sha256": _SHA256,
        "model_id": _NONEMPTY_STRING,
        "beliefs": {
            "type": "object",
            "required": ["attribute_1", "attribute_2", "attribute_3"],
            "additionalProperties": False,
            "properties": {
                attribute: _PROBABILITY_ROW
                for attribute in (
                    "attribute_1",
                    "attribute_2",
                    "attribute_3",
                )
            },
        },
        "raw_response_sha256": {
            "anyOf": [_SHA256, {"type": "null"}]
        },
    },
    "additionalProperties": False,
}


_OPTION = {
    "type": "object",
    "required": ["option_id", "features", "label", "domain"],
    "additionalProperties": False,
    "properties": {
        "option_id": {"type": "string", "minLength": 1},
        "features": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
        "label": {"type": "string"},
        "domain": {"type": "string"},
    },
}

_CONTEXT = {
    "type": "object",
    "required": [
        "context_id",
        "domain",
        "scenario_id",
        "turn_id",
        "options",
        "ranking",
        "default",
        "suggested_option",
        "wording_template",
        "question_type",
        "target_attribute",
    ],
    "additionalProperties": False,
    "properties": {
        "context_id": {"type": "string", "minLength": 1},
        "domain": {"type": "string"},
        "scenario_id": {"type": "string"},
        "turn_id": {"type": "string"},
        "options": {"type": "array", "minItems": 1, "items": _OPTION},
        "ranking": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "default": {"type": ["string", "null"]},
        "suggested_option": {"type": ["string", "null"]},
        "wording_template": {"type": "string", "minLength": 1},
        "question_type": {"type": "string", "minLength": 1},
        "target_attribute": {
            "anyOf": [
                {"type": "integer", "minimum": 0, "maximum": 2},
                {"type": "null"},
            ]
        },
    },
}

_PROVENANCE = {
    "type": "object",
    "required": [
        "policy_id",
        "policy_version",
        "profile_snapshot",
        "random_seed",
        "config_digest",
        "presentation_mechanism",
        "profile_conditioned",
    ],
    "additionalProperties": False,
    "properties": {
        "policy_id": {"type": "string", "minLength": 1},
        "policy_version": {"type": "string", "minLength": 1},
        "profile_snapshot": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "random_seed": {"type": "integer"},
        "config_digest": {"type": "string"},
        "presentation_mechanism": {
            "enum": [
                "none",
                "balanced",
                "ranking",
                "default",
                "suggestion",
                "restriction",
                "target_selection",
            ]
        },
        "profile_conditioned": {"type": "boolean"},
    },
}

_OBSERVATION = {
    "type": "object",
    "required": ["selected_option", "surface_response", "choice_noise_key"],
    "additionalProperties": False,
    "properties": {
        "selected_option": {"type": "string", "minLength": 1},
        "surface_response": {"type": ["string", "null"]},
        "choice_noise_key": {"type": "string"},
    },
}

_PROFILE_UPDATE = {
    "type": "object",
    "required": [
        "updater_id",
        "belief_before",
        "belief_after",
        "native_memory_before",
        "native_memory_after",
        "written_delta",
    ],
    "additionalProperties": False,
    "properties": {
        "updater_id": {"type": "string", "minLength": 1},
        "belief_before": {"type": "array", "items": {"type": "number", "minimum": 0}},
        "belief_after": {"type": "array", "items": {"type": "number", "minimum": 0}},
        "native_memory_before": {"type": "array", "items": {"type": "string"}},
        "native_memory_after": {"type": "array", "items": {"type": "string"}},
        "written_delta": {"type": "array", "items": {"type": "string"}},
    },
}

_INTERACTION_RECORD = {
    "type": "object",
    "required": [
        "record_id",
        "context",
        "policy_provenance",
        "observation",
        "profile_update",
    ],
    "additionalProperties": False,
    "properties": {
        "record_id": {"type": "string", "minLength": 1},
        "context": _CONTEXT,
        "policy_provenance": _PROVENANCE,
        "observation": _OBSERVATION,
        "profile_update": {
            "anyOf": [_PROFILE_UPDATE, {"type": "null"}]
        },
    },
}

_TERMINAL_ACTION = {
    "type": "object",
    "required": [
        "item_id",
        "item_sha256",
        "wording_template_id",
        "question_type",
        "selected_option_id",
        "declared_direction",
    ],
    "additionalProperties": False,
    "properties": {
        "item_id": _NONEMPTY_STRING,
        "item_sha256": _SHA256,
        "wording_template_id": _NONEMPTY_STRING,
        "question_type": {
            "enum": [
                "forced_choice",
                "counterfactual_choice",
                "direct_preference_probe",
                "cross_context_choice",
            ]
        },
        "selected_option_id": {"type": ["string", "null"]},
        "declared_direction": {
            "anyOf": [{"enum": [-1, 1]}, {"type": "null"}]
        },
    },
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "user-state": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:user-state:v1",
        "title": "CAPE-Loop latent user state",
        "type": "object",
        "required": ["schema_version", "user_id", "domain", "theta", "susceptibility", "split"],
        "properties": {
            "schema_version": {"const": 1},
            "user_id": {"type": "string", "minLength": 1},
            "domain": {"enum": ["travel", "writing"]},
            "theta": {
                "type": "array",
                "prefixItems": [{"enum": [-2, -1, 1, 2]}] * 3,
                "minItems": 3,
                "maxItems": 3,
            },
            "susceptibility": {
                "type": "object",
                "required": ["ranking", "default", "suggestion"],
                "additionalProperties": False,
                "properties": {
                    "ranking": {"type": "number", "minimum": 0},
                    "default": {"type": "number", "minimum": 0},
                    "suggestion": {"type": "number", "minimum": 0},
                },
            },
            "split": {"enum": ["train", "development", "test"]},
        },
        "additionalProperties": False,
    },
    "llm-response": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:llm-response:v1",
        "title": "CAPE-Loop LLM structured profile response",
        **_LLM_RESPONSE_RECORD,
    },
    "llm-request": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:llm-request:v1",
        "title": "CAPE-Loop provider-neutral LLM request",
        "type": "object",
        "required": [
            "schema_version",
            "request_id",
            "updater_id",
            "view",
            "system_instruction",
            "payload",
            "prompt_sha256",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "request_id": {"type": "string", "minLength": 1},
            "updater_id": {"type": "string", "minLength": 1},
            "view": {
                "enum": ["response_only", "full_context", "provenance_aware"]
            },
            "system_instruction": {"type": "string", "minLength": 1},
            "payload": {"type": "object"},
            "prompt_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
    },
    "interaction-record": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:interaction-record:v1",
        "title": "CAPE-Loop causal interaction record",
        "description": (
            "The visible context and internal policy provenance are sibling "
            "records so their causal roles cannot be conflated."
        ),
        **_INTERACTION_RECORD,
    },
    "trajectory": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:trajectory:v1",
        "title": "CAPE-Loop auditable trajectory",
        "$defs": {
            # Keep the trajectory schema standalone. The separately exported
            # interaction-record schema remains useful as its own public
            # contract, but consumers need no custom URN registry here.
            "interaction_record": _INTERACTION_RECORD,
        },
        "type": "object",
        "required": ["trajectory_id", "user_id", "domain", "interactions"],
        "additionalProperties": False,
        "properties": {
            "trajectory_id": {"type": "string", "minLength": 1},
            "user_id": {"type": "string", "minLength": 1},
            "domain": {"enum": ["travel", "writing"]},
            "interactions": {
                "type": "array",
                "items": {"$ref": "#/$defs/interaction_record"},
            },
        },
    },
    "run-manifest": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:run-manifest:v1",
        "title": "CAPE-Loop run manifest",
        "type": "object",
        "required": [
            "schema_version",
            "run_id",
            "config_sha256",
            "config_origin",
            "git_revision",
            "source_sha256",
            "deterministic",
            "status",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "run_id": {"type": "string", "minLength": 1},
            "config_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "config_origin": {
                "oneOf": [
                    {
                        "type": "object",
                        "required": [
                            "kind",
                            "descriptor",
                            "config_sha256",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "kind": {"const": "programmatic"},
                            "descriptor": _NONEMPTY_STRING,
                            "config_sha256": _SHA256,
                        },
                    },
                    {
                        "type": "object",
                        "required": [
                            "kind",
                            "retained_file",
                            "source_filename",
                            "source_sha256",
                            "config_sha256",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "kind": {"const": "toml_file"},
                            "retained_file": {
                                "const": "config.source.toml"
                            },
                            "source_filename": _NONEMPTY_STRING,
                            "source_sha256": _SHA256,
                            "config_sha256": _SHA256,
                        },
                    },
                ]
            },
            "git_revision": {"type": ["string", "null"]},
            "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "deterministic": {"type": "boolean"},
            "status": {"enum": ["created", "complete", "failed"]},
        },
    },
    "human-rating": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:human-rating:v1",
        "title": "CAPE-Loop blinded pragmatic-evidence rating",
        "type": "object",
        "required": ["assignment_id", "display_id", "rating"],
        "additionalProperties": False,
        "properties": {
            "assignment_id": {"type": "string", "minLength": 1},
            "display_id": {"type": "string", "minLength": 1},
            "rating": {"type": "integer", "minimum": 1, "maximum": 7},
        },
    },
    "external-decoder-request": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:external-decoder-request:v1",
        "title": "CAPE-Loop blinded external decoder request",
        "description": (
            "A content-addressed request payload for an external decoder. "
            "Runtime parsing additionally rejects recursively nested leakage "
            "keys and verifies request_sha256."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "request_id",
            "pseudonymous_state_id",
            "representation_id",
            "evaluation_split",
            "rubric_version",
            "payload",
            "instruction",
            "request_sha256",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "request_id": _NONEMPTY_STRING,
            "pseudonymous_state_id": _NONEMPTY_STRING,
            "representation_id": _NONEMPTY_STRING,
            "evaluation_split": _SPLIT,
            "rubric_version": _NONEMPTY_STRING,
            "payload": {
                "type": "object",
                "minProperties": 1,
            },
            "instruction": _NONEMPTY_STRING,
            "request_sha256": _SHA256,
        },
    },
    "external-decoder-judgment": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:external-decoder-judgment:v1",
        "title": "CAPE-Loop external decoder judgment",
        "description": (
            "A model- or human-supplied decoder judgment. This exchange "
            "record does not assert that any judgment has been collected or "
            "that distinct sources are statistically independent."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "request_id",
            "request_sha256",
            "decoder_instance_id",
            "decoder_family_id",
            "judgment_origin",
            "source_descriptor",
            "blind_to_system_identity",
            "blind_to_latent_truth",
            "probabilities",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "request_id": _NONEMPTY_STRING,
            "request_sha256": _SHA256,
            "decoder_instance_id": _NONEMPTY_STRING,
            "decoder_family_id": _NONEMPTY_STRING,
            "judgment_origin": {
                "enum": ["external_model", "human_annotator"]
            },
            "source_descriptor": _NONEMPTY_STRING,
            "blind_to_system_identity": {"type": "boolean"},
            "blind_to_latent_truth": {"type": "boolean"},
            "probabilities": _PROBABILITY_ROWS,
        },
    },
    "decoder-truth-label": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:decoder-truth-label:v1",
        "title": "CAPE-Loop separately retained decoder truth label",
        "description": (
            "A latent truth row joined to blinded decoder requests only after "
            "external judgments are collected."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "pseudonymous_state_id",
            "theta",
            "evaluation_split",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "pseudonymous_state_id": _NONEMPTY_STRING,
            "theta": {
                "type": "array",
                "items": {"enum": [-2, -1, 1, 2]},
                "minItems": 3,
                "maxItems": 3,
            },
            "evaluation_split": _SPLIT,
        },
    },
    "native-terminal-action-record": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:native-terminal-action-record:v1",
        "title": "CAPE-Loop recorded native terminal action record",
        "description": (
            "A content-addressed recorded live/replay action set. Runtime "
            "validation binds it to a verified Experiment B trajectory and "
            "the exact retained held-out terminal suite."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "record_id",
            "trajectory_id",
            "domain_id",
            "updater_id",
            "evaluation_split",
            "adapter_kind",
            "evidence_origin",
            "native_state_id",
            "native_system_id",
            "native_system_version",
            "suite_id",
            "suite_sha256",
            "action_execution_mode",
            "execution_trace_sha256",
            "recorded_at",
            "recording_attestation",
            "actions",
            "record_sha256",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "record_id": _NONEMPTY_STRING,
            "trajectory_id": _NONEMPTY_STRING,
            "domain_id": _NONEMPTY_STRING,
            "updater_id": {
                "enum": [
                    "episodic_memory",
                    "semantic_memory",
                    "provenance_linked_memory",
                ]
            },
            "evaluation_split": {"const": "test"},
            "adapter_kind": {"const": "native_end_to_end_recorded"},
            "evidence_origin": {"const": "imported_native_system"},
            "native_state_id": _SHA256,
            "native_system_id": _NONEMPTY_STRING,
            "native_system_version": _NONEMPTY_STRING,
            "suite_id": _NONEMPTY_STRING,
            "suite_sha256": _SHA256,
            "action_execution_mode": {
                "enum": ["recorded_live", "recorded_replay"]
            },
            "execution_trace_sha256": _SHA256,
            "recorded_at": {"type": "string", "format": "date-time"},
            "recording_attestation": {
                "const": (
                    "actions_emitted_by_named_native_system_not_"
                    "reference_projection"
                )
            },
            "actions": {
                "type": "array",
                "minItems": 1,
                "items": _TERMINAL_ACTION,
            },
            "record_sha256": _SHA256,
        },
    },
    "decoder-source-review": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:decoder-source-review:v1",
        "title": "CAPE-Loop responsible-researcher decoder source review",
        "description": (
            "A hash-bound eligibility determination. It records review of "
            "source dependencies and does not itself prove independent errors."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "review_id",
            "responsible_researcher_id",
            "reviewed_at",
            "requests_sha256",
            "judgments_sha256",
            "decision",
            "source_assessments",
            "pair_assessments",
            "review_sha256",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "review_id": _NONEMPTY_STRING,
            "responsible_researcher_id": _NONEMPTY_STRING,
            "reviewed_at": {"type": "string", "format": "date-time"},
            "requests_sha256": _SHA256,
            "judgments_sha256": _SHA256,
            "decision": {
                "enum": ["eligible_distinct_sources", "not_eligible"]
            },
            "source_assessments": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "decoder_instance_id",
                        "decoder_family_id",
                        "judgment_origin",
                        "source_descriptor",
                        "eligible_for_gate4",
                        "dependency_notes",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "decoder_instance_id": _NONEMPTY_STRING,
                        "decoder_family_id": _NONEMPTY_STRING,
                        "judgment_origin": {
                            "enum": ["external_model", "human_annotator"]
                        },
                        "source_descriptor": _NONEMPTY_STRING,
                        "eligible_for_gate4": {"type": "boolean"},
                        "dependency_notes": _NONEMPTY_STRING,
                    },
                },
            },
            "pair_assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "left_decoder_instance_id",
                        "right_decoder_instance_id",
                        "genuinely_distinct_for_claimed_scope",
                        "rationale",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "left_decoder_instance_id": _NONEMPTY_STRING,
                        "right_decoder_instance_id": _NONEMPTY_STRING,
                        "genuinely_distinct_for_claimed_scope": {
                            "type": "boolean"
                        },
                        "rationale": _NONEMPTY_STRING,
                    },
                },
            },
            "review_sha256": _SHA256,
        },
    },
    "gate4-review-artifact": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:gate4-review-artifact:v1",
        "title": "CAPE-Loop immutable Gate 4 evidence review",
        "type": "object",
        "required": [
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "claim_status",
            "source_run",
            "inputs",
            "validation_summary",
            "gate_4",
            "interpretation_boundary",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "artifact_kind": {"const": "gate4-native-evidence-review"},
            "artifact_id": _SHA256,
            "claim_status": {"const": "not_claimed"},
            "source_run": {
                "type": "object",
                "required": [
                    "run_id",
                    "run_manifest_sha256",
                    "run_checksum_manifest_sha256",
                    "config_sha256",
                    "source_sha256",
                    "verified_complete",
                ],
                "additionalProperties": False,
                "properties": {
                    "run_id": _NONEMPTY_STRING,
                    "run_manifest_sha256": _SHA256,
                    "run_checksum_manifest_sha256": _SHA256,
                    "config_sha256": _SHA256,
                    "source_sha256": _SHA256,
                    "verified_complete": {"const": True},
                },
            },
            "inputs": {
                "type": "object",
                "required": [
                    "decoder_requests",
                    "decoder_judgments",
                    "decoder_truth_labels",
                    "native_terminal_actions",
                    "decoder_source_review",
                ],
                "additionalProperties": False,
                "properties": {
                    name: {
                        "type": "object",
                        "required": ["filename", "sha256", "bytes"],
                        "properties": {
                            "filename": _NONEMPTY_STRING,
                            "sha256": _SHA256,
                            "bytes": {"type": "integer", "minimum": 0},
                            "record_count": {
                                "type": "integer",
                                "minimum": 1,
                            },
                        },
                        "additionalProperties": False,
                    }
                    for name in (
                        "decoder_requests",
                        "decoder_judgments",
                        "decoder_truth_labels",
                        "native_terminal_actions",
                        "decoder_source_review",
                    )
                },
            },
            "validation_summary": {"type": "object"},
            "gate_4": {"type": "object"},
            "interpretation_boundary": _NONEMPTY_STRING,
        },
    },
    "human-collection": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:human-collection:v1",
        "title": "CAPE-Loop de-identified human collection record",
        "description": (
            "A collection contract carrying consent, blinding, and "
            "comprehension metadata. It makes no claim of ethics approval or "
            "completed recruitment."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "participant_code",
            "assignment_id",
            "assignment_protocol_id",
            "display_id",
            "rating",
            "response_time_ms",
            "consent_version",
            "consented",
            "blinding_version",
            "comprehension_check_id",
            "comprehension_passed",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "participant_code": _NONEMPTY_STRING,
            "assignment_id": _NONEMPTY_STRING,
            "assignment_protocol_id": _NONEMPTY_STRING,
            "display_id": _NONEMPTY_STRING,
            "rating": {"type": "integer", "minimum": 1, "maximum": 7},
            "response_time_ms": {"type": "integer", "minimum": 0},
            "consent_version": _NONEMPTY_STRING,
            "consented": {"type": "boolean"},
            "blinding_version": _NONEMPTY_STRING,
            "comprehension_check_id": _NONEMPTY_STRING,
            "comprehension_passed": {"type": "boolean"},
        },
    },
    "heldout-paraphrase-case": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:heldout-paraphrase-case:v1",
        "title": "CAPE-Loop held-out surface paraphrase case",
        "type": "object",
        "required": [
            "schema_version",
            "case_id",
            "source_trial_id",
            "domain_id",
            "mechanism",
            "selected_option_id",
            "template_id",
            "family_id",
            "split",
            "template_sha256",
            "context_sha256",
            "surface_response",
            "binding_sha256",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "case_id": _NONEMPTY_STRING,
            "source_trial_id": _NONEMPTY_STRING,
            "domain_id": _NONEMPTY_STRING,
            "mechanism": _MECHANISM,
            "selected_option_id": _NONEMPTY_STRING,
            "template_id": _NONEMPTY_STRING,
            "family_id": _NONEMPTY_STRING,
            "split": _SPLIT,
            "template_sha256": _SHA256,
            "context_sha256": _SHA256,
            "surface_response": _NONEMPTY_STRING,
            "binding_sha256": _SHA256,
        },
    },
    "heldout-paraphrase-evaluation": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:heldout-paraphrase-evaluation:v1",
        "title": "CAPE-Loop held-out paraphrase updater evaluation",
        "type": "object",
        "required": [
            "schema_version",
            "case_id",
            "binding_sha256",
            "source_trial_id",
            "template_id",
            "family_id",
            "split",
            "domain_id",
            "mechanism",
            "updater_id",
            "brier",
            "belief_sha256",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "case_id": _NONEMPTY_STRING,
            "binding_sha256": _SHA256,
            "source_trial_id": _NONEMPTY_STRING,
            "template_id": _NONEMPTY_STRING,
            "family_id": _NONEMPTY_STRING,
            "split": _SPLIT,
            "domain_id": _NONEMPTY_STRING,
            "mechanism": _MECHANISM,
            "updater_id": _NONEMPTY_STRING,
            "brier": {"type": "number", "minimum": 0, "maximum": 2},
            "belief_sha256": _SHA256,
        },
    },
    "heldout-paraphrase-criterion": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:heldout-paraphrase-criterion:v1",
        "title": "CAPE-Loop Gate 1 held-out paraphrase criterion",
        "description": (
            "A nullable verification result: null means the required external "
            "or paired evaluation records are incomplete."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "criterion_id",
            "verified",
            "complete",
            "material_gap",
            "required_mechanisms",
            "covered_domains",
            "covered_template_ids",
            "expected_template_ids",
            "qualifying_mechanisms",
            "mean_gaps",
            "missing_pairs",
            "gate_1_argument",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "criterion_id": {"const": "held-out-paraphrase-transfer"},
            "verified": {"type": ["boolean", "null"]},
            "complete": {"type": "boolean"},
            "material_gap": {"type": "number", "minimum": 0},
            "required_mechanisms": {"type": "integer", "minimum": 1},
            "covered_domains": {
                "type": "array",
                "items": _NONEMPTY_STRING,
                "uniqueItems": True,
            },
            "covered_template_ids": {
                "type": "array",
                "items": _NONEMPTY_STRING,
                "uniqueItems": True,
            },
            "expected_template_ids": {
                "type": "array",
                "items": _NONEMPTY_STRING,
                "uniqueItems": True,
            },
            "qualifying_mechanisms": {
                "type": "array",
                "items": _MECHANISM,
                "uniqueItems": True,
            },
            "mean_gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "family_id",
                        "domain_id",
                        "mechanism",
                        "full_context_minus_aware_brier",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "family_id": _NONEMPTY_STRING,
                        "domain_id": _NONEMPTY_STRING,
                        "mechanism": _MECHANISM,
                        "full_context_minus_aware_brier": {
                            "type": "number"
                        },
                    },
                },
            },
            "missing_pairs": {
                "type": "array",
                "items": _NONEMPTY_STRING,
                "uniqueItems": True,
            },
            "gate_1_argument": {"type": ["boolean", "null"]},
        },
    },
    "openai-provider-audit": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:openai-provider-audit:v1",
        "title": "CAPE-Loop OpenAI Responses API audit record",
        "description": (
            "A provider execution sidecar bound to a provider-neutral replay "
            "response. Its acceptance status records whether a completed "
            "attempt may enter replay; the schema itself does not claim that "
            "an evaluation was run."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "provider",
            "acceptance_status",
            "request_id",
            "prompt_sha256",
            "request_body_sha256",
            "model_requested",
            "model_returned",
            "provider_response_id",
            "provider_created_at",
            "usage",
            "started_at",
            "completed_at",
            "attempts",
            "idempotency_key",
            "client_request_id",
            "server_request_id",
            "processing_ms",
            "estimated_max_tokens",
            "raw_response_sha256",
            "raw_response",
            "replay_response",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "provider": {"const": "openai"},
            "acceptance_status": {
                "enum": [
                    "accepted",
                    "rejected_model_mismatch",
                ]
            },
            "request_id": _NONEMPTY_STRING,
            "prompt_sha256": _SHA256,
            "request_body_sha256": _SHA256,
            "model_requested": _NONEMPTY_STRING,
            "model_returned": _NONEMPTY_STRING,
            "provider_response_id": _NONEMPTY_STRING,
            "provider_created_at": {"type": ["number", "null"]},
            "usage": {"type": "object"},
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time"},
            "attempts": {"type": "integer", "minimum": 1},
            "idempotency_key": _NONEMPTY_STRING,
            "client_request_id": _NONEMPTY_STRING,
            "server_request_id": {"type": ["string", "null"]},
            "processing_ms": {"type": ["string", "null"]},
            "estimated_max_tokens": {"type": "integer", "minimum": 1},
            "raw_response_sha256": {
                "anyOf": [_SHA256, {"type": "null"}]
            },
            "raw_response": {"type": "object"},
            "replay_response": _LLM_RESPONSE_RECORD,
        },
    },
}


def export_schemas(destination: str | Path) -> tuple[Path, ...]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for name, schema in sorted(SCHEMAS.items()):
        path = root / f"{name}.schema.json"
        path.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(path)
    return tuple(written)
