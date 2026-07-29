from __future__ import annotations

from types import SimpleNamespace
import unittest

from cape_loop.conversation_reporting import (
    conversation_stats,
    render_markdown,
    select_diverse_records,
    updater_model_ids,
)


def _record(
    identifier: str,
    *,
    domain: str,
    mechanism: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_kind": "conversation_trace",
        "experiment": "A",
        "conversation_id": identifier,
        "conversation_kind": "single_turn",
        "source_id": identifier,
        "user_id": "test-user",
        "domain_id": domain,
        "conditions": {
            "split": "test",
            "mechanism": mechanism,
        },
        "dialogue": [
            {
                "turn": 1,
                "event_id": identifier,
                "scenario_id": "scenario-1",
                "surface_id": "surface-1",
                "surface_available": True,
                "assistant": "Here are Hotel A and Hotel B.",
                "user": "I choose Hotel A.",
                "selected_option_id": "hotel-a",
                "selected_option_label": "Hotel A",
                "presentation_mechanism": mechanism,
                "choice_source": "mathematical_user_simulator",
                "surface_source": "frozen test bank",
                "turn_metrics": {},
            }
        ],
        "outcomes": [
            {
                "updater_id": "llm_full_context",
                "updater_view": "full_context",
                "model_ids": ["provider/model"],
                "metrics": {"acue": 0.125},
            }
        ],
        "assessments": [],
        "comparisons": [],
    }


class ConversationReportingTests(unittest.TestCase):
    def test_stats_and_markdown_explain_roles_and_complete_counts(self) -> None:
        records = (
            _record("a-1", domain="travel", mechanism="balanced"),
            _record("a-2", domain="writing", mechanism="default"),
        )
        stats = conversation_stats(records)
        self.assertEqual(
            stats,
            {
                "record_count": 2,
                "turn_count": 2,
                "outcome_count": 2,
            },
        )
        markdown = render_markdown(
            records[:1],
            experiment="A",
            complete_stats=stats,
            complete_jsonl_path="conversations/experiment-a.jsonl",
            records_are_preselected=True,
        )
        self.assertIn("preview of **1** of **2**", markdown)
        self.assertIn("Scenario presenter (assistant)", markdown)
        self.assertIn("Simulated user", markdown)
        self.assertIn("Evaluated profile updater", markdown)
        self.assertIn("I choose Hotel A.", markdown)
        self.assertIn("provider/model", markdown)
        self.assertIn("ACUE", markdown)

    def test_preview_round_robins_across_condition_groups(self) -> None:
        records = tuple(
            [
                _record(
                    f"balanced-{index}",
                    domain="travel",
                    mechanism="balanced",
                )
                for index in range(5)
            ]
            + [
                _record(
                    "default-1",
                    domain="travel",
                    mechanism="default",
                )
            ]
        )
        preview = select_diverse_records(records, limit=2)
        self.assertEqual(
            {record["conditions"]["mechanism"] for record in preview},
            {"balanced", "default"},
        )
        self.assertEqual(
            select_diverse_records(records, limit=2),
            preview,
        )

    def test_model_ids_are_taken_from_observed_responses(self) -> None:
        registry = {
            "llm_full_context": SimpleNamespace(
                responses=(
                    SimpleNamespace(model_id="provider/model-b"),
                    SimpleNamespace(model_id="provider/model-a"),
                    SimpleNamespace(model_id="provider/model-b"),
                )
            ),
            "exact_action_aware": SimpleNamespace(responses=()),
        }
        self.assertEqual(
            updater_model_ids(registry),
            {
                "llm_full_context": (
                    "provider/model-a",
                    "provider/model-b",
                ),
                "exact_action_aware": (),
            },
        )


if __name__ == "__main__":
    unittest.main()
