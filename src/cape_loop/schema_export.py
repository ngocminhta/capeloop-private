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

_TERMINAL_BATTERY_SCORE = {
    "type": "object",
    "required": [
        "profile_brier",
        "behavioral_accuracy",
        "tie_excluded_behavioral_accuracy",
        "fractional_behavioral_accuracy",
        "cross_context_accuracy",
        "mean_intrinsic_regret",
        "predicted_option_ids",
        "predicted_utility_tie_count",
        "intrinsic_utility_tie_count",
        "evaluated_item_count",
        "profile_ece",
        "profile_calibration_sample_unit",
        "profile_calibration_prediction_count",
        "profile_reliability_bins",
        "profile_calibration_interpretation",
    ],
    "additionalProperties": False,
    "properties": {
        "profile_brier": {"type": "number"},
        "behavioral_accuracy": {"type": "number"},
        "tie_excluded_behavioral_accuracy": {
            "type": ["number", "null"]
        },
        "fractional_behavioral_accuracy": {"type": "number"},
        "cross_context_accuracy": {"type": ["number", "null"]},
        "mean_intrinsic_regret": {"type": "number"},
        "predicted_option_ids": {
            "type": "array",
            "items": _NONEMPTY_STRING,
        },
        "predicted_utility_tie_count": {"type": "number"},
        "intrinsic_utility_tie_count": {"type": "number"},
        "evaluated_item_count": {"type": "integer", "minimum": 1},
        "profile_ece": {"type": ["number", "null"]},
        "profile_calibration_sample_unit": {
            "const": "preference_attribute_forecast"
        },
        "profile_calibration_prediction_count": {
            "type": "integer",
            "minimum": 0,
        },
        "profile_reliability_bins": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "bin_index",
                    "lower",
                    "upper",
                    "prediction_count",
                    "mean_confidence",
                    "empirical_accuracy",
                ],
                "additionalProperties": False,
                "properties": {
                    "bin_index": {"type": "integer", "minimum": 0},
                    "lower": {"type": "number"},
                    "upper": {"type": "number"},
                    "prediction_count": {
                        "type": "integer",
                        "minimum": 0,
                    },
                    "mean_confidence": {
                        "type": ["number", "null"]
                    },
                    "empirical_accuracy": {
                        "type": ["number", "null"]
                    },
                },
            },
        },
        "profile_calibration_interpretation": _NONEMPTY_STRING,
    },
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

_SCENARIO_OPTION = {
    "type": "object",
    "required": ["option_id", "label", "features"],
    "additionalProperties": False,
    "properties": {
        "option_id": _NONEMPTY_STRING,
        "label": _NONEMPTY_STRING,
        "features": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
}

_SCENARIO_RECORD = {
    "type": "object",
    "required": [
        "scenario_id",
        "family_id",
        "revision",
        "status",
        "split",
        "domain",
        "task_family",
        "target_attribute",
        "target_key",
        "difficulty",
        "prompt",
        "wording_template_id",
        "negative_option",
        "positive_option",
        "negative_same_direction_option",
        "positive_same_direction_option",
        "supported_mechanisms",
        "quality_assertions",
        "review",
    ],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {
                "properties": {"status": {"const": "approved"}},
                "required": ["status"],
            },
            "then": {
                "properties": {
                    "review": {
                        "properties": {
                            "automated_validation": {"const": "passed"},
                            "surface_human_review": {"const": "passed"},
                            "scientific_human_review": {"const": "passed"},
                        }
                    }
                }
            },
        }
    ],
    "properties": {
        "scenario_id": _NONEMPTY_STRING,
        "family_id": _NONEMPTY_STRING,
        "revision": {"type": "integer", "minimum": 1},
        "status": {"enum": ["provisional", "approved"]},
        "split": _SPLIT,
        "domain": {"enum": ["travel", "writing"]},
        "task_family": _NONEMPTY_STRING,
        "target_attribute": {
            "type": "integer",
            "minimum": 0,
            "maximum": 2,
        },
        "target_key": _NONEMPTY_STRING,
        "difficulty": {
            "enum": ["standard_tradeoff", "close_tradeoff"]
        },
        "prompt": _NONEMPTY_STRING,
        "wording_template_id": _NONEMPTY_STRING,
        "negative_option": _SCENARIO_OPTION,
        "positive_option": _SCENARIO_OPTION,
        "negative_same_direction_option": _SCENARIO_OPTION,
        "positive_same_direction_option": _SCENARIO_OPTION,
        "supported_mechanisms": {
            "type": "array",
            "uniqueItems": True,
            "minItems": 6,
            "maxItems": 6,
            "items": {
                "enum": [
                    "balanced",
                    "restricted",
                    "default",
                    "suggested",
                    "ranking",
                    "suggestion",
                ]
            },
        },
        "quality_assertions": {
            "type": "object",
            "required": [
                "neutral_wording",
                "symmetric_surface",
                "no_treatment_cues",
                "no_split_cues",
                "no_real_entities",
                "no_time_sensitive_facts",
                "no_objective_dominance",
                "all_surface_facts_modeled_or_matched",
                "feature_role_contract",
            ],
            "additionalProperties": False,
            "properties": {
                key: {"const": True}
                for key in (
                    "neutral_wording",
                    "symmetric_surface",
                    "no_treatment_cues",
                    "no_split_cues",
                    "no_real_entities",
                    "no_time_sensitive_facts",
                    "no_objective_dominance",
                    "all_surface_facts_modeled_or_matched",
                    "feature_role_contract",
                )
            },
        },
        "review": {
            "type": "object",
            "required": [
                "automated_validation",
                "surface_human_review",
                "scientific_human_review",
                "paper_eligible",
                "note",
            ],
            "additionalProperties": False,
            "properties": {
                "automated_validation": {"enum": ["pending", "passed"]},
                "surface_human_review": {
                    "enum": ["not_completed", "passed"]
                },
                "scientific_human_review": {
                    "enum": ["not_completed", "passed"]
                },
                "paper_eligible": {"const": False},
                "note": _NONEMPTY_STRING,
            },
        },
    },
}

_SCENARIO_CATALOG_INPUT = {
    "type": "object",
    "required": [
        "schema_version",
        "input_kind",
        "catalog_id",
        "catalog_version",
        "catalog_status",
        "eligibility",
        "selection_policy",
        "source_filename",
        "source_sha256",
        "retained_file",
        "scenario_count",
        "family_count",
        "paper_eligible",
    ],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": 1},
        "input_kind": {"const": "scenario_catalog"},
        "catalog_id": _NONEMPTY_STRING,
        "catalog_version": _NONEMPTY_STRING,
        "catalog_status": {"const": "frozen-development"},
        "eligibility": {"const": "simulation-and-pilot-only"},
        "selection_policy": {
            "const": "deterministic-stratified-v1"
        },
        "source_filename": _NONEMPTY_STRING,
        "source_sha256": _SHA256,
        "retained_file": {"const": "inputs/scenario-catalog.json"},
        "scenario_count": {"type": "integer", "minimum": 1},
        "family_count": {"type": "integer", "minimum": 1},
        "paper_eligible": {"const": False},
    },
}

_CONVERSATION_TEMPLATE_INPUT = {
    "type": "object",
    "required": [
        "schema_version",
        "input_kind",
        "bank_id",
        "source",
        "scenario_count",
        "retained_file",
    ],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": 1},
        "input_kind": {"const": "conversation_template_bank"},
        "bank_id": _NONEMPTY_STRING,
        "source": _NONEMPTY_STRING,
        "scenario_count": {"type": "integer", "minimum": 1},
        "retained_file": {
            "const": "inputs/conversation-templates.json"
        },
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
        "prompt": {"type": "string", "minLength": 1, "maxLength": 500},
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
    "required": [
        "selected_option",
        "surface_response",
        "choice_noise_key",
        "assistant_message",
        "surface_id",
    ],
    "additionalProperties": False,
    "properties": {
        "selected_option": {"type": "string", "minLength": 1},
        "surface_response": {"type": ["string", "null"]},
        "choice_noise_key": {"type": "string"},
        "assistant_message": {"type": ["string", "null"]},
        "surface_id": {"type": "string"},
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

_CONVERSATION_METRIC_VALUE = {
    "anyOf": [
        {"type": "number"},
        {"type": "boolean"},
        {"type": "string"},
        {"type": "null"},
        {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
        },
    ]
}

_CONVERSATION_DIALOGUE_TURN = {
    "type": "object",
    "required": [
        "turn",
        "event_id",
        "scenario_id",
        "surface_id",
        "surface_available",
        "assistant",
        "user",
        "selected_option_id",
        "selected_option_label",
        "presentation_mechanism",
        "choice_source",
        "surface_source",
        "turn_metrics",
    ],
    "additionalProperties": False,
    "properties": {
        "turn": {"type": "integer", "minimum": 1},
        "event_id": _NONEMPTY_STRING,
        "scenario_id": {
            "anyOf": [_NONEMPTY_STRING, {"type": "null"}]
        },
        "surface_id": {
            "anyOf": [_NONEMPTY_STRING, {"type": "null"}]
        },
        "surface_available": {"type": "boolean"},
        "assistant": {
            "anyOf": [_NONEMPTY_STRING, {"type": "null"}]
        },
        "user": {
            "anyOf": [_NONEMPTY_STRING, {"type": "null"}]
        },
        "selected_option_id": _NONEMPTY_STRING,
        "selected_option_label": {
            "anyOf": [_NONEMPTY_STRING, {"type": "null"}]
        },
        "presentation_mechanism": _NONEMPTY_STRING,
        "choice_source": {"const": "mathematical_user_simulator"},
        "surface_source": {
            "anyOf": [_NONEMPTY_STRING, {"type": "null"}]
        },
        "turn_metrics": {
            "type": "object",
            "additionalProperties": _CONVERSATION_METRIC_VALUE,
        },
    },
}

_CONVERSATION_OUTCOME = {
    "type": "object",
    "required": [
        "updater_id",
        "updater_view",
        "model_ids",
        "metrics",
    ],
    "additionalProperties": False,
    "properties": {
        "updater_id": _NONEMPTY_STRING,
        "updater_view": {
            "anyOf": [_NONEMPTY_STRING, {"type": "null"}]
        },
        "model_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": _NONEMPTY_STRING,
        },
        "metrics": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": _CONVERSATION_METRIC_VALUE,
        },
    },
}

_CONVERSATION_ASSESSMENT = {
    "type": "object",
    "required": ["attribute", "metrics", "clause_results"],
    "additionalProperties": False,
    "properties": {
        "attribute": {
            "type": "integer",
            "minimum": 0,
            "maximum": 2,
        },
        "metrics": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": _CONVERSATION_METRIC_VALUE,
        },
        "clause_results": {
            "type": "object",
            "additionalProperties": {"type": "boolean"},
        },
    },
}

_CONVERSATION_COMPARISON = {
    "type": "object",
    "required": ["comparison_id", "metrics"],
    "additionalProperties": False,
    "properties": {
        "comparison_id": _NONEMPTY_STRING,
        "metrics": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": _CONVERSATION_METRIC_VALUE,
        },
    },
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "scenario-catalog": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:scenario-catalog:v1",
        "title": "CAPE-Loop versioned scenario catalog",
        "type": "object",
        "required": [
            "schema_version",
            "catalog_id",
            "catalog_version",
            "catalog_status",
            "eligibility",
            "language",
            "locale",
            "source",
            "license",
            "created_on",
            "frozen_on",
            "split_policy",
            "selection_policy",
            "attribute_order",
            "authoring_provenance",
            "scenarios",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "catalog_id": _NONEMPTY_STRING,
            "catalog_version": _NONEMPTY_STRING,
            "catalog_status": {"const": "frozen-development"},
            "eligibility": {"const": "simulation-and-pilot-only"},
            "language": _NONEMPTY_STRING,
            "locale": _NONEMPTY_STRING,
            "source": {"const": "project-authored-synthetic"},
            "license": _NONEMPTY_STRING,
            "created_on": _NONEMPTY_STRING,
            "frozen_on": _NONEMPTY_STRING,
            "split_policy": {"const": "scenario-family-disjoint-v1"},
            "selection_policy": {
                "const": "deterministic-stratified-v1"
            },
            "attribute_order": {
                "type": "object",
                "required": ["travel", "writing"],
                "additionalProperties": False,
                "properties": {
                    "travel": {
                        "const": ["price", "setting", "planning"]
                    },
                    "writing": {
                        "const": ["length", "tone", "spelling"]
                    },
                },
            },
            "authoring_provenance": {
                "type": "object",
                "minProperties": 1,
            },
            "scenarios": {
                "type": "array",
                "minItems": 1,
                "items": _SCENARIO_RECORD,
            },
        },
    },
    "conversation-template-bank": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:conversation-template-bank:v1",
        "title": "CAPE-Loop frozen conversation template bank",
        "type": "object",
        "required": [
            "schema_version",
            "bank_id",
            "source",
            "templates",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "bank_id": _NONEMPTY_STRING,
            "source": _NONEMPTY_STRING,
            "templates": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "scenario_id",
                        "display_names",
                        "presentation_templates",
                        "choice_template",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "scenario_id": _NONEMPTY_STRING,
                        "display_names": {
                            "type": "object",
                            "minProperties": 4,
                            "maxProperties": 4,
                            "additionalProperties": _NONEMPTY_STRING,
                        },
                        "presentation_templates": {
                            "type": "object",
                            "required": [
                                "balanced",
                                "restricted",
                                "default",
                                "suggested",
                                "ranking",
                            ],
                            "additionalProperties": False,
                            "properties": {
                                name: _NONEMPTY_STRING
                                for name in (
                                    "balanced",
                                    "restricted",
                                    "default",
                                    "suggested",
                                    "ranking",
                                )
                            },
                        },
                        "choice_template": _NONEMPTY_STRING,
                        "source": _NONEMPTY_STRING,
                    },
                },
            },
        },
    },
    "conversation-log": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:conversation-log:v1",
        "title": "CAPE-Loop natural-language conversation and metric trace",
        "description": (
            "A compact reporting view that stores visible dialogue once and "
            "groups the evaluated updater outcomes beside it. It intentionally "
            "excludes latent user state, option feature vectors, beliefs, "
            "native memory, and provider prompts or responses."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "record_kind",
            "experiment",
            "conversation_id",
            "conversation_kind",
            "source_id",
            "user_id",
            "domain_id",
            "conditions",
            "dialogue",
            "outcomes",
            "assessments",
            "comparisons",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "record_kind": {"const": "conversation_trace"},
            "experiment": {
                "enum": ["A", "B", "C", "sensitivity", "demo"]
            },
            "conversation_id": _NONEMPTY_STRING,
            "conversation_kind": {
                "enum": [
                    "single_turn",
                    "closed_loop",
                    "fixed_history",
                ]
            },
            "source_id": _NONEMPTY_STRING,
            "user_id": _NONEMPTY_STRING,
            "domain_id": {"enum": ["travel", "writing"]},
            "conditions": {
                "type": "object",
                "additionalProperties": _CONVERSATION_METRIC_VALUE,
            },
            "dialogue": {
                "type": "array",
                "minItems": 1,
                "items": _CONVERSATION_DIALOGUE_TURN,
            },
            "outcomes": {
                "type": "array",
                "minItems": 1,
                "items": _CONVERSATION_OUTCOME,
            },
            "assessments": {
                "type": "array",
                "items": _CONVERSATION_ASSESSMENT,
            },
            "comparisons": {
                "type": "array",
                "items": _CONVERSATION_COMPARISON,
            },
        },
    },
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
            "inputs": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "scenario_catalog": _SCENARIO_CATALOG_INPUT,
                    "conversation_templates": (
                        _CONVERSATION_TEMPLATE_INPUT
                    ),
                },
            },
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
    "experiment-c-decoder-codebook": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:experiment-c-decoder-codebook:v1",
        "title": "CAPE-Loop Experiment C external-decoder codebook row",
        "description": (
            "A researcher-only binding from one blinded native terminal state "
            "to an exact content-addressed Experiment C metric and state row."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "request_id",
            "pseudonymous_state_id",
            "evaluation_split",
            "regime",
            "replicate",
            "user_id",
            "domain_id",
            "updater_id",
            "stable_row_key_sha256",
            "source_metric_row_sha256",
            "battery_id",
            "battery_digest",
            "terminal_state_id",
            "terminal_state_sha256",
            "source_state_file",
            "source_state_record_id",
            "source_state_record_sha256",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "request_id": _NONEMPTY_STRING,
            "pseudonymous_state_id": _NONEMPTY_STRING,
            "evaluation_split": {"enum": ["development", "test"]},
            "regime": {
                "enum": [
                    "fixed_balanced",
                    "fixed_biased",
                    "endogenous_closed_loop",
                ]
            },
            "replicate": {"type": "integer", "minimum": 0},
            "user_id": _NONEMPTY_STRING,
            "domain_id": _NONEMPTY_STRING,
            "updater_id": {
                "enum": [
                    "episodic_memory",
                    "semantic_memory",
                    "provenance_linked_memory",
                ]
            },
            "stable_row_key_sha256": _SHA256,
            "source_metric_row_sha256": _SHA256,
            "battery_id": _NONEMPTY_STRING,
            "battery_digest": _SHA256,
            "terminal_state_id": _SHA256,
            "terminal_state_sha256": _SHA256,
            "source_state_file": {
                "enum": [
                    "events/experiment-c-replays.jsonl",
                    "events/experiment-c-endogenous.jsonl",
                ]
            },
            "source_state_record_id": _NONEMPTY_STRING,
            "source_state_record_sha256": _SHA256,
        },
    },
    "experiment-c-external-score": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:experiment-c-external-score:v1",
        "title": "CAPE-Loop calibrated Experiment C external decoder score",
        "description": (
            "One development-calibrated external-family belief scored on the "
            "exact common terminal battery. Exactly two such rows are averaged "
            "for each eligible native Experiment C metric row."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "source_metric_row_sha256",
            "stable_row_key_sha256",
            "request_id",
            "request_sha256",
            "pseudonymous_state_id",
            "evaluation_split",
            "regime",
            "replicate",
            "user_id",
            "domain_id",
            "updater_id",
            "battery_id",
            "battery_digest",
            "decoder_instance_id",
            "decoder_family_id",
            "source_descriptor",
            "judgment_origin",
            "blind_to_system_identity",
            "blind_to_latent_truth",
            "calibration_fitted_split",
            "calibration_temperature",
            "calibration_example_count",
            "calibrated_marginals",
            "terminal_score",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "source_metric_row_sha256": _SHA256,
            "stable_row_key_sha256": _SHA256,
            "request_id": _NONEMPTY_STRING,
            "request_sha256": _SHA256,
            "pseudonymous_state_id": _NONEMPTY_STRING,
            "evaluation_split": {"enum": ["development", "test"]},
            "regime": {
                "enum": [
                    "fixed_balanced",
                    "fixed_biased",
                    "endogenous_closed_loop",
                ]
            },
            "replicate": {"type": "integer", "minimum": 0},
            "user_id": _NONEMPTY_STRING,
            "domain_id": _NONEMPTY_STRING,
            "updater_id": {
                "enum": [
                    "episodic_memory",
                    "semantic_memory",
                    "provenance_linked_memory",
                ]
            },
            "battery_id": _NONEMPTY_STRING,
            "battery_digest": _SHA256,
            "decoder_instance_id": _NONEMPTY_STRING,
            "decoder_family_id": _NONEMPTY_STRING,
            "source_descriptor": _NONEMPTY_STRING,
            "judgment_origin": {"const": "external_model"},
            "blind_to_system_identity": {"const": True},
            "blind_to_latent_truth": {"const": True},
            "calibration_fitted_split": {"const": "development"},
            "calibration_temperature": {
                "type": "number",
                "exclusiveMinimum": 0,
            },
            "calibration_example_count": {
                "type": "integer",
                "minimum": 1,
            },
            "calibrated_marginals": _PROBABILITY_ROWS,
            "terminal_score": _TERMINAL_BATTERY_SCORE,
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
                    "native_collection_plan",
                    "native_action_requests",
                    "native_transport_attempts",
                    "native_provider_audit",
                    "native_terminal_actions",
                    "native_execution_manifest",
                    "decoder_source_review",
                ],
                "oneOf": [
                    {
                        "required": [
                            "decoder_collection_plan",
                            "decoder_transport_attempts",
                            "decoder_provider_audit",
                            "decoder_execution_manifest",
                        ]
                    },
                    {
                        "not": {
                            "anyOf": [
                                {"required": [name]}
                                for name in (
                                    "decoder_collection_plan",
                                    "decoder_transport_attempts",
                                    "decoder_provider_audit",
                                    "decoder_execution_manifest",
                                )
                            ]
                        }
                    },
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
                        "decoder_collection_plan",
                        "decoder_transport_attempts",
                        "decoder_provider_audit",
                        "decoder_judgments",
                        "decoder_execution_manifest",
                        "decoder_truth_labels",
                        "native_collection_plan",
                        "native_action_requests",
                        "native_transport_attempts",
                        "native_provider_audit",
                        "native_terminal_actions",
                        "native_execution_manifest",
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
    "human-model-evidence": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:human-model-evidence:v1",
        "title": "CAPE-Loop H8 model evidence-strength record",
        "description": (
            "One held-out fitted-aware or LLM evidence-strength observation "
            "for the H8 human-versus-model comparison. Runtime validation also "
            "requires stable role/metric/artifact metadata within a source."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "source_run_id",
            "source_artifact_sha256",
            "source_record_id",
            "source_id",
            "source_role",
            "cluster_id",
            "scenario_id",
            "condition",
            "evidence_strength",
            "evidence_metric",
            "zero_means_no_evidence",
            "evaluation_split",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "source_run_id": _NONEMPTY_STRING,
            "source_artifact_sha256": _SHA256,
            "source_record_id": _NONEMPTY_STRING,
            "source_id": _NONEMPTY_STRING,
            "source_role": {
                "enum": [
                    "fitted_action_aware",
                    "ordinary_llm",
                    "provenance_aware_llm",
                ]
            },
            "cluster_id": _NONEMPTY_STRING,
            "scenario_id": _NONEMPTY_STRING,
            "condition": {
                "enum": [
                    "volunteered",
                    "balanced",
                    "restricted",
                    "default",
                    "suggested",
                ]
            },
            "evidence_strength": {"type": "number", "minimum": 0},
            "evidence_metric": {
                "const": "positive_part_anchor_directional_log_odds_update"
            },
            "zero_means_no_evidence": {"const": True},
            "evaluation_split": {"const": "test"},
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
    "openrouter-provider-audit": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:cape-loop:schema:openrouter-provider-audit:v1",
        "title": "CAPE-Loop OpenRouter Chat Completions audit record",
        "description": (
            "A gateway execution sidecar retaining the requested OpenRouter "
            "model, selected upstream route, additive router metadata, and "
            "the provider-neutral replay response. It does not claim direct "
            "first-party provider origin."
        ),
        "type": "object",
        "required": [
            "schema_version",
            "provider",
            "gateway",
            "acceptance_status",
            "request_id",
            "prompt_sha256",
            "request_body_sha256",
            "model_requested",
            "model_returned",
            "upstream_provider",
            "upstream_model",
            "routing_strategy",
            "routing_attempt",
            "routing_metadata",
            "provider_response_id",
            "provider_created_at",
            "usage",
            "started_at",
            "completed_at",
            "transport_attempts",
            "attempts",
            "client_request_id",
            "generation_id",
            "cache_status",
            "estimated_max_tokens",
            "upstream_provider_constraint",
            "provider_preferences",
            "route_constraint_evidence",
            "selected_upstream_identity_semantics",
            "raw_response_sha256",
            "raw_response",
            "replay_response",
            "first_party_origin_claimed",
        ],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 1},
            "provider": {"const": "openrouter"},
            "gateway": {"const": "openrouter"},
            "acceptance_status": {
                "enum": [
                    "accepted",
                    "rejected_openrouter_identity",
                ]
            },
            "request_id": _NONEMPTY_STRING,
            "prompt_sha256": _SHA256,
            "request_body_sha256": _SHA256,
            "model_requested": _NONEMPTY_STRING,
            "model_returned": _NONEMPTY_STRING,
            "upstream_provider": _NONEMPTY_STRING,
            "upstream_model": _NONEMPTY_STRING,
            "routing_strategy": _NONEMPTY_STRING,
            "routing_attempt": {"type": "integer", "minimum": 1},
            "routing_metadata": {"type": "object"},
            "provider_response_id": _NONEMPTY_STRING,
            "provider_created_at": {"type": ["number", "null"]},
            "usage": {"type": "object"},
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time"},
            "transport_attempts": {"type": "integer", "minimum": 1},
            "attempts": {"type": "integer", "minimum": 1},
            "client_request_id": _NONEMPTY_STRING,
            "generation_id": {"type": ["string", "null"]},
            "cache_status": {"type": ["string", "null"]},
            "estimated_max_tokens": {"type": "integer", "minimum": 1},
            "upstream_provider_constraint": {
                "type": ["string", "null"]
            },
            "provider_preferences": {"type": "object"},
            "route_constraint_evidence": {
                "enum": [
                    "request_body_provider_only_and_order",
                    (
                        "request_body_provider_preferences_without_exact_"
                        "route_constraint"
                    ),
                ]
            },
            "selected_upstream_identity_semantics": {
                "const": (
                    "router_display_identity_not_exact_route_slug_attestation"
                )
            },
            "raw_response_sha256": {
                "anyOf": [_SHA256, {"type": "null"}]
            },
            "raw_response": {"type": "object"},
            "replay_response": _LLM_RESPONSE_RECORD,
            "first_party_origin_claimed": {"const": False},
        },
    },
}


def _embedded_record(schema_name: str) -> dict[str, Any]:
    """Return a standalone record schema without document-level metadata."""

    return {
        key: value
        for key, value in SCHEMAS[schema_name].items()
        if key not in {"$schema", "$id", "title", "description"}
    }


SCHEMAS.update(
    {
        "llm-provider-transport-attempt": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": (
                "urn:cape-loop:schema:"
                "llm-provider-transport-attempt:v1"
            ),
            "title": "CAPE-Loop direct LLM provider transport attempt",
            "description": (
                "One fsynced started or settled event around a physical "
                "OpenAI/OpenRouter HTTP attempt. Final settlements embed the "
                "accepted/rejected provider audit so a paid response can be "
                "recovered after process failure."
            ),
            "oneOf": [
                {
                    "type": "object",
                    "required": [
                        "schema_version",
                        "kind",
                        "event",
                        "attempt_id",
                        "provider",
                        "request_id",
                        "prompt_sha256",
                        "endpoint",
                        "request_body_sha256",
                        "model_requested",
                        "client_request_id",
                        "idempotency_key",
                        "estimated_max_tokens",
                        "attempt_ordinal",
                        "started_at",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "schema_version": {"const": 1},
                        "kind": {
                            "const": "llm-provider-transport-attempt"
                        },
                        "event": {"const": "started"},
                        "attempt_id": _SHA256,
                        "provider": {
                            "enum": ["openai", "openrouter"]
                        },
                        "request_id": _NONEMPTY_STRING,
                        "prompt_sha256": _SHA256,
                        "endpoint": {
                            "type": "string",
                            "format": "uri",
                            "minLength": 1,
                        },
                        "request_body_sha256": _SHA256,
                        "model_requested": _NONEMPTY_STRING,
                        "client_request_id": _NONEMPTY_STRING,
                        "idempotency_key": {
                            "anyOf": [
                                _NONEMPTY_STRING,
                                {"type": "null"},
                            ]
                        },
                        "estimated_max_tokens": {
                            "type": "integer",
                            "minimum": 1,
                        },
                        "attempt_ordinal": {
                            "type": "integer",
                            "minimum": 1,
                        },
                        "started_at": {
                            "type": "string",
                            "format": "date-time",
                        },
                    },
                },
                {
                    "type": "object",
                    "required": [
                        "schema_version",
                        "kind",
                        "event",
                        "attempt_id",
                        "settled_at",
                        "outcome",
                        "automatic_retry_safe",
                        "http_status",
                        "charged_tokens",
                        "server_request_id",
                        "response_body_sha256",
                        "response_record",
                        "provider_audit",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "schema_version": {"const": 1},
                        "kind": {
                            "const": "llm-provider-transport-attempt"
                        },
                        "event": {"const": "settled"},
                        "attempt_id": _SHA256,
                        "settled_at": {
                            "type": "string",
                            "format": "date-time",
                        },
                        "outcome": {
                            "enum": [
                                "transport_error",
                                "http_error",
                                "invalid_response",
                                "rejected_provider_result",
                                "success",
                            ]
                        },
                        "automatic_retry_safe": {"type": "boolean"},
                        "http_status": {
                            "type": ["integer", "null"],
                            "minimum": 100,
                            "maximum": 599,
                        },
                        "charged_tokens": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "server_request_id": {
                            "type": ["string", "null"]
                        },
                        "response_body_sha256": {
                            "anyOf": [_SHA256, {"type": "null"}]
                        },
                        "response_record": {
                            "anyOf": [
                                {"type": "object"},
                                {"type": "null"},
                            ]
                        },
                        "provider_audit": {
                            "anyOf": [
                                _embedded_record(
                                    "openai-provider-audit"
                                ),
                                _embedded_record(
                                    "openrouter-provider-audit"
                                ),
                                {"type": "null"},
                            ]
                        },
                    },
                },
            ],
        }
    }
)


SCHEMAS.update(
    {
        "external-decoder-provider-audit": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": (
                "urn:cape-loop:schema:"
                "external-decoder-provider-audit:v1"
            ),
            "title": "CAPE-Loop external decoder provider audit",
            "description": (
                "An audit-first Anthropic or Google Gemini response record "
                "with an embedded import-compatible decoder judgment."
            ),
            "type": "object",
            "required": [
                "schema_version",
                "kind",
                "acceptance_status",
                "provider",
                "request_id",
                "request_sha256",
                "prompt_sha256",
                "decoder_instance_id",
                "decoder_family_id",
                "source_descriptor",
                "request_body_sha256",
                "model_requested",
                "model_returned",
                "provider_response_id",
                "usage",
                "started_at",
                "completed_at",
                "attempts",
                "client_request_id",
                "server_request_id",
                "estimated_max_tokens",
                "judgment",
                "llm_response",
                "raw_response",
            ],
            "additionalProperties": False,
            "properties": {
                "schema_version": {"const": 1},
                "kind": {"const": "external-decoder-provider-audit"},
                "acceptance_status": {
                    "enum": [
                        "accepted",
                        "rejected_identity_mismatch",
                    ]
                },
                "provider": {
                    "enum": ["anthropic", "google_gemini"]
                },
                "request_id": _NONEMPTY_STRING,
                "request_sha256": _SHA256,
                "prompt_sha256": _SHA256,
                "decoder_instance_id": _NONEMPTY_STRING,
                "decoder_family_id": _NONEMPTY_STRING,
                "source_descriptor": _NONEMPTY_STRING,
                "request_body_sha256": _SHA256,
                "model_requested": _NONEMPTY_STRING,
                "model_returned": _NONEMPTY_STRING,
                "provider_response_id": _NONEMPTY_STRING,
                "usage": {"type": "object"},
                "started_at": {"type": "string", "format": "date-time"},
                "completed_at": {"type": "string", "format": "date-time"},
                "attempts": {"type": "integer", "minimum": 1},
                "client_request_id": _NONEMPTY_STRING,
                "server_request_id": {"type": ["string", "null"]},
                "estimated_max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                },
                "judgment": _embedded_record(
                    "external-decoder-judgment"
                ),
                "llm_response": _LLM_RESPONSE_RECORD,
                "raw_response": {"type": "object"},
            },
        },
        "native-action-provider-audit": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:cape-loop:schema:native-action-provider-audit:v1",
            "title": "CAPE-Loop OpenAI native-action provider audit",
            "description": (
                "An audit-first Responses API record whose embedded action "
                "record is bound to retained native memory and a held-out "
                "terminal suite."
            ),
            "type": "object",
            "required": [
                "schema_version",
                "provider",
                "workflow",
                "acceptance_status",
                "request_id",
                "trajectory_id",
                "prompt_sha256",
                "native_state_id",
                "suite_sha256",
                "request_body_sha256",
                "model_requested",
                "model_returned",
                "provider_response_id",
                "usage",
                "started_at",
                "completed_at",
                "attempts",
                "idempotency_key",
                "client_request_id",
                "server_request_id",
                "estimated_max_tokens",
                "raw_response_sha256",
                "raw_response",
                "action_record",
            ],
            "additionalProperties": False,
            "properties": {
                "schema_version": {"const": 1},
                "provider": {"const": "openai"},
                "workflow": {"const": "native_terminal_actions"},
                "acceptance_status": {
                    "enum": [
                        "accepted",
                        "rejected_model_mismatch",
                    ]
                },
                "request_id": _NONEMPTY_STRING,
                "trajectory_id": _NONEMPTY_STRING,
                "prompt_sha256": _SHA256,
                "native_state_id": _SHA256,
                "suite_sha256": _SHA256,
                "request_body_sha256": _SHA256,
                "model_requested": _NONEMPTY_STRING,
                "model_returned": _NONEMPTY_STRING,
                "provider_response_id": _NONEMPTY_STRING,
                "usage": {"type": "object"},
                "started_at": {"type": "string", "format": "date-time"},
                "completed_at": {"type": "string", "format": "date-time"},
                "attempts": {"type": "integer", "minimum": 1},
                "idempotency_key": _NONEMPTY_STRING,
                "client_request_id": _NONEMPTY_STRING,
                "server_request_id": {"type": ["string", "null"]},
                "estimated_max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                },
                "raw_response_sha256": _SHA256,
                "raw_response": {"type": "object"},
                "action_record": _embedded_record(
                    "native-terminal-action-record"
                ),
            },
        },
    }
)

SCHEMAS.update(
    {
        "external-decoder-transport-attempt": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": (
                "urn:cape-loop:schema:"
                "external-decoder-transport-attempt:v1"
            ),
            "title": "CAPE-Loop external decoder transport attempt",
            "description": (
                "A durable started or settled physical HTTP-attempt event. "
                "The Python reader additionally enforces ordering, digest "
                "bindings, and outcome/audit consistency."
            ),
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "schema_version",
                        "kind",
                        "event",
                        "attempt_id",
                        "provider",
                        "request_id",
                        "request_sha256",
                        "prompt_sha256",
                        "decoder_instance_id",
                        "request_body_sha256",
                        "model_requested",
                        "client_request_id",
                        "estimated_max_tokens",
                        "attempt_ordinal",
                        "started_at",
                    ],
                    "properties": {
                        "schema_version": {"const": 1},
                        "kind": {
                            "const": "external-decoder-transport-attempt"
                        },
                        "event": {"const": "started"},
                        "attempt_id": _SHA256,
                        "provider": {
                            "enum": ["anthropic", "google_gemini"]
                        },
                        "request_id": _NONEMPTY_STRING,
                        "request_sha256": _SHA256,
                        "prompt_sha256": _SHA256,
                        "decoder_instance_id": _NONEMPTY_STRING,
                        "request_body_sha256": _SHA256,
                        "model_requested": _NONEMPTY_STRING,
                        "client_request_id": _NONEMPTY_STRING,
                        "estimated_max_tokens": {
                            "type": "integer",
                            "minimum": 1,
                        },
                        "attempt_ordinal": {
                            "type": "integer",
                            "minimum": 1,
                        },
                        "started_at": {
                            "type": "string",
                            "format": "date-time",
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "schema_version",
                        "kind",
                        "event",
                        "attempt_id",
                        "settled_at",
                        "outcome",
                        "http_status",
                        "charged_tokens",
                        "server_request_id",
                        "response_body_sha256",
                        "provider_audit",
                    ],
                    "properties": {
                        "schema_version": {"const": 1},
                        "kind": {
                            "const": "external-decoder-transport-attempt"
                        },
                        "event": {"const": "settled"},
                        "attempt_id": _SHA256,
                        "settled_at": {
                            "type": "string",
                            "format": "date-time",
                        },
                        "outcome": {
                            "enum": [
                                "transport_error",
                                "http_error",
                                "invalid_provider_metadata",
                                "invalid_response",
                                "identity_mismatch",
                                "success",
                            ]
                        },
                        "http_status": {
                            "type": ["integer", "null"],
                            "minimum": 100,
                            "maximum": 599,
                        },
                        "charged_tokens": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "server_request_id": {
                            "type": ["string", "null"]
                        },
                        "response_body_sha256": {
                            "anyOf": [_SHA256, {"type": "null"}]
                        },
                        "provider_audit": {
                            "anyOf": [
                                _embedded_record(
                                    "external-decoder-provider-audit"
                                ),
                                {"type": "null"},
                            ]
                        },
                    },
                },
            ],
        },
        "native-action-transport-attempt": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:cape-loop:schema:native-action-transport-attempt:v1",
            "title": "CAPE-Loop native-action transport attempt",
            "description": (
                "A durable started or settled physical OpenAI HTTP-attempt "
                "event, bound to the reviewed native-action collection plan."
            ),
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "schema_version",
                        "kind",
                        "event",
                        "attempt_id",
                        "collection_plan_sha256",
                        "collection_config_sha256",
                        "request_id",
                        "prompt_sha256",
                        "native_state_id",
                        "suite_sha256",
                        "request_body_sha256",
                        "model_requested",
                        "idempotency_key",
                        "client_request_id",
                        "estimated_max_tokens",
                        "attempt_ordinal",
                        "started_at",
                    ],
                    "properties": {
                        "schema_version": {"const": 1},
                        "kind": {
                            "const": "native-action-transport-attempt"
                        },
                        "event": {"const": "started"},
                        "attempt_id": _SHA256,
                        "collection_plan_sha256": _SHA256,
                        "collection_config_sha256": _SHA256,
                        "request_id": _NONEMPTY_STRING,
                        "prompt_sha256": _SHA256,
                        "native_state_id": _SHA256,
                        "suite_sha256": _SHA256,
                        "request_body_sha256": _SHA256,
                        "model_requested": _NONEMPTY_STRING,
                        "idempotency_key": _NONEMPTY_STRING,
                        "client_request_id": _NONEMPTY_STRING,
                        "estimated_max_tokens": {
                            "type": "integer",
                            "minimum": 1,
                        },
                        "attempt_ordinal": {
                            "type": "integer",
                            "minimum": 1,
                        },
                        "started_at": {
                            "type": "string",
                            "format": "date-time",
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "schema_version",
                        "kind",
                        "event",
                        "attempt_id",
                        "settled_at",
                        "outcome",
                        "http_status",
                        "charged_tokens",
                        "server_request_id",
                        "response_body_sha256",
                        "response_record",
                        "provider_audit",
                    ],
                    "properties": {
                        "schema_version": {"const": 1},
                        "kind": {
                            "const": "native-action-transport-attempt"
                        },
                        "event": {"const": "settled"},
                        "attempt_id": _SHA256,
                        "settled_at": {
                            "type": "string",
                            "format": "date-time",
                        },
                        "outcome": {
                            "enum": [
                                "transport_error",
                                "http_error",
                                "invalid_response",
                                "model_mismatch",
                                "success",
                            ]
                        },
                        "http_status": {
                            "type": ["integer", "null"],
                            "minimum": 100,
                            "maximum": 599,
                        },
                        "charged_tokens": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "server_request_id": {
                            "type": ["string", "null"]
                        },
                        "response_body_sha256": {
                            "anyOf": [_SHA256, {"type": "null"}]
                        },
                        "response_record": {
                            "anyOf": [
                                {"type": "object"},
                                {"type": "null"},
                            ]
                        },
                        "provider_audit": {
                            "anyOf": [
                                _embedded_record(
                                    "native-action-provider-audit"
                                ),
                                {"type": "null"},
                            ]
                        },
                    },
                },
            ],
        },
    }
)


_H7_SOURCE_RUN_BINDING = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "run_id",
        "manifest_sha256",
        "config_file_sha256",
        "checksums_sha256",
        "population_sha256",
        "experiment_a_metrics_sha256",
        "hypothesis_estimands_sha256",
    ],
    "properties": {
        "run_id": _NONEMPTY_STRING,
        "manifest_sha256": _SHA256,
        "config_file_sha256": _SHA256,
        "checksums_sha256": _SHA256,
        "population_sha256": _SHA256,
        "experiment_a_metrics_sha256": _SHA256,
        "hypothesis_estimands_sha256": _SHA256,
    },
}

_H7_PROBABILITY_BELIEFS = {
    "type": "object",
    "additionalProperties": False,
    "required": ["attribute_1", "attribute_2", "attribute_3"],
    "properties": {
        attribute: _PROBABILITY_ROW
        for attribute in ("attribute_1", "attribute_2", "attribute_3")
    },
}

_H7_CASE = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "case_id",
        "user_id",
        "domain_id",
        "target_attribute",
        "target_direction",
        "surface_statement",
        "source_user_sha256",
        "prior_probabilities",
        "case_sha256",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "case_id": _NONEMPTY_STRING,
        "user_id": _NONEMPTY_STRING,
        "domain_id": {"enum": ["travel", "writing"]},
        "target_attribute": {
            "type": "integer",
            "minimum": 0,
            "maximum": 2,
        },
        "target_direction": {"enum": [-1, 1]},
        "surface_statement": _NONEMPTY_STRING,
        "source_user_sha256": _SHA256,
        "prior_probabilities": _H7_PROBABILITY_BELIEFS,
        "case_sha256": _SHA256,
    },
}

_H7_REQUEST_BINDING = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "case_id",
        "case_sha256",
        "user_id",
        "updater_id",
        "view",
        "llm_request",
        "binding_sha256",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "case_id": _NONEMPTY_STRING,
        "case_sha256": _SHA256,
        "user_id": _NONEMPTY_STRING,
        "updater_id": {
            "enum": ["llm_full_context", "llm_provenance_aware"]
        },
        "view": {"enum": ["full_context", "provenance_aware"]},
        "llm_request": _embedded_record("llm-request"),
        "binding_sha256": _SHA256,
    },
}

_H7_VOLUNTEERED_EVIDENCE = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "case_id",
        "user_id",
        "domain_id",
        "target_attribute",
        "target_direction",
        "updater_id",
        "provider",
        "model_id",
        "request_id",
        "prompt_sha256",
        "request_body_sha256",
        "raw_response_sha256",
        "audit_record_sha256",
        "prior_probabilities",
        "posterior_probabilities",
        "directional_log_odds_update",
        "claim_status",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "case_id": _NONEMPTY_STRING,
        "user_id": _NONEMPTY_STRING,
        "domain_id": {"enum": ["travel", "writing"]},
        "target_attribute": {
            "type": "integer",
            "minimum": 0,
            "maximum": 2,
        },
        "target_direction": {"enum": [-1, 1]},
        "updater_id": {
            "enum": ["llm_full_context", "llm_provenance_aware"]
        },
        "provider": {"enum": ["openai", "openrouter"]},
        "model_id": _NONEMPTY_STRING,
        "request_id": _NONEMPTY_STRING,
        "prompt_sha256": _SHA256,
        "request_body_sha256": _SHA256,
        "raw_response_sha256": _SHA256,
        "audit_record_sha256": _SHA256,
        "prior_probabilities": _H7_PROBABILITY_BELIEFS,
        "posterior_probabilities": _H7_PROBABILITY_BELIEFS,
        "directional_log_odds_update": {"type": "number"},
        "claim_status": {"const": "not_claimed"},
    },
}

SCHEMAS.update(
    {
        "h7-volunteered-request-binding": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": (
                "urn:cape-loop:schema:"
                "h7-volunteered-request-binding:v1"
            ),
            "title": "CAPE-Loop H7 volunteered request binding",
            "description": (
                "A provider-neutral direct-statement request plus the "
                "withheld source-user, case, updater, and view bindings."
            ),
            **_H7_REQUEST_BINDING,
        },
        "h7-volunteered-collection-plan": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": (
                "urn:cape-loop:schema:"
                "h7-volunteered-collection-plan:v1"
            ),
            "title": "CAPE-Loop H7 volunteered collection plan",
            "description": (
                "The complete content-addressed direct-statement corpus "
                "derived from one verified Experiment A run."
            ),
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "artifact_kind",
                "plan_version",
                "source_run",
                "roles",
                "cases",
                "request_bindings",
                "case_count",
                "request_count",
                "independent_user_count",
                "plan_sha256",
                "claim_status",
                "interpretation",
            ],
            "properties": {
                "schema_version": {"const": 1},
                "artifact_kind": {
                    "const": "h7_volunteered_collection_plan"
                },
                "plan_version": {
                    "const": "h7-volunteered-control-plan-v1"
                },
                "source_run": _H7_SOURCE_RUN_BINDING,
                "roles": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["updater_id", "view"],
                        "properties": {
                            "updater_id": {
                                "enum": [
                                    "llm_full_context",
                                    "llm_provenance_aware",
                                ]
                            },
                            "view": {
                                "enum": [
                                    "full_context",
                                    "provenance_aware",
                                ]
                            },
                        },
                    },
                },
                "cases": {
                    "type": "array",
                    "minItems": 1,
                    "items": _H7_CASE,
                },
                "request_bindings": {
                    "type": "array",
                    "minItems": 2,
                    "items": _H7_REQUEST_BINDING,
                },
                "case_count": {"type": "integer", "minimum": 1},
                "request_count": {"type": "integer", "minimum": 2},
                "independent_user_count": {
                    "type": "integer",
                    "minimum": 2,
                },
                "plan_sha256": _SHA256,
                "claim_status": {"const": "not_claimed"},
                "interpretation": _NONEMPTY_STRING,
            },
        },
        "h7-volunteered-evidence": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:cape-loop:schema:h7-volunteered-evidence:v1",
            "title": "CAPE-Loop H7 provider-bound volunteered evidence",
            "description": (
                "One accepted direct-statement provider response and its "
                "directional log-odds update."
            ),
            **_H7_VOLUNTEERED_EVIDENCE,
        },
        "h7-volunteered-review": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:cape-loop:schema:h7-volunteered-review:v1",
            "title": "CAPE-Loop H7 volunteered-control review",
            "description": (
                "A derived, checksum-bound review that supplies H7's "
                "volunteered positive-control component without modifying "
                "the source run."
            ),
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "artifact_kind",
                "review_version",
                "source_run",
                "collection_plan",
                "provider_evidence",
                "analysis_settings",
                "volunteered_preference_updates",
                "provider_bound_evidence",
                "source_h7_sha256",
                "recomputed_h7",
                "recomputation_scope",
                "claim_status",
                "interpretation",
                "review_sha256",
            ],
            "properties": {
                "schema_version": {"const": 1},
                "artifact_kind": {
                    "const": "h7_volunteered_control_review"
                },
                "review_version": {
                    "const": "h7-volunteered-control-review-v1"
                },
                "source_run": _H7_SOURCE_RUN_BINDING,
                "collection_plan": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "plan_sha256",
                        "plan_file_sha256",
                        "bindings_file_sha256",
                        "requests_file_sha256",
                    ],
                    "properties": {
                        "plan_sha256": _SHA256,
                        "plan_file_sha256": _SHA256,
                        "bindings_file_sha256": _SHA256,
                        "requests_file_sha256": _SHA256,
                    },
                },
                "provider_evidence": {"type": "object"},
                "analysis_settings": {"type": "object"},
                "volunteered_preference_updates": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "case_id",
                            "user_id",
                            "updater_id",
                            "directional_log_odds_update",
                        ],
                        "properties": {
                            "case_id": _NONEMPTY_STRING,
                            "user_id": _NONEMPTY_STRING,
                            "updater_id": {
                                "enum": [
                                    "llm_full_context",
                                    "llm_provenance_aware",
                                ]
                            },
                            "directional_log_odds_update": {
                                "type": "number"
                            },
                        },
                    },
                },
                "provider_bound_evidence": {
                    "type": "array",
                    "minItems": 2,
                    "items": _H7_VOLUNTEERED_EVIDENCE,
                },
                "source_h7_sha256": _SHA256,
                "recomputed_h7": {"type": "object"},
                "recomputation_scope": {"type": "object"},
                "claim_status": {"const": "not_claimed"},
                "interpretation": _NONEMPTY_STRING,
                "review_sha256": _SHA256,
            },
        },
    }
)


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
