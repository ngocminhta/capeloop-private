from __future__ import annotations

import json
import unittest

from cape_loop.beliefs import PreferenceBelief
from cape_loop.domains import TRAVEL
from cape_loop.elicitation import build_matched_anchor_set
from cape_loop.native import (
    EpisodicMemoryUpdater,
    NativeMemoryState,
    ProvenanceLinkedMemoryUpdater,
    SemanticMemoryUpdater,
    blinded_decoder_views,
    decode_native_state,
)
from cape_loop.schemas import PolicyProvenance
from cape_loop.updaters import UpdateViewKind, make_update_view


def event_fixture(mechanism: str = "suggested"):
    matched = build_matched_anchor_set(
        TRAVEL,
        target_attribute=0,
        anchor_direction=-1,
        scenario_id="native-fixture",
    )
    context = matched.context(mechanism)
    observation = matched.observation()
    provenance = PolicyProvenance(
        "soft_profile_conditioned",
        "v1",
        (
            ("attribute_1", -1.25),
            ("attribute_2", 0.0),
            ("attribute_3", 0.0),
        ),
        random_seed=7,
    )
    return context, observation, provenance


class NativeMemoryTests(unittest.TestCase):
    def test_episdodic_semantic_and_provenance_linked_states_are_auditable(
        self,
    ) -> None:
        prior = PreferenceBelief.uniform()
        context, observation, provenance = event_fixture()

        episodic = EpisodicMemoryUpdater()
        episodic_view = make_update_view(
            UpdateViewKind.FULL_CONTEXT,
            context,
            observation,
            provenance,
            event_id="native-event",
        )
        episodic_result = episodic.update(
            episodic.initial_state(prior),
            episodic_view,
        )
        episodic_state = episodic_result.state.opaque_state
        self.assertIsInstance(episodic_state, NativeMemoryState)
        assert isinstance(episodic_state, NativeMemoryState)
        self.assertEqual(episodic_state.memory_kind, "episodic")
        self.assertEqual(len(episodic_state.episodes), 1)
        self.assertEqual(episodic_state.claims, ())
        self.assertEqual(
            episodic_result.profile_update.native_memory_after[0],
            episodic_state.state_id,
        )

        semantic = SemanticMemoryUpdater()
        semantic_result = semantic.update(
            semantic.initial_state(prior),
            episodic_view,
        )
        semantic_state = semantic_result.state.opaque_state
        self.assertIsInstance(semantic_state, NativeMemoryState)
        assert isinstance(semantic_state, NativeMemoryState)
        self.assertEqual(semantic_state.memory_kind, "semantic")
        self.assertEqual(
            semantic_state.claims[0].source_event_ids,
            ("native-event",),
        )
        self.assertNotEqual(
            semantic_state.persona_belief,
            episodic_state.persona_belief,
        )

        linked = ProvenanceLinkedMemoryUpdater()
        linked_view = make_update_view(
            UpdateViewKind.PROVENANCE_AWARE,
            context,
            observation,
            provenance,
            event_id="native-event",
        )
        linked_result = linked.update(
            linked.initial_state(prior),
            linked_view,
        )
        linked_state = linked_result.state.opaque_state
        self.assertIsInstance(linked_state, NativeMemoryState)
        assert isinstance(linked_state, NativeMemoryState)
        self.assertEqual(linked_state.memory_kind, "provenance_linked")
        self.assertEqual(
            linked_state.episodes[0].provenance_policy_id,
            "soft_profile_conditioned",
        )
        self.assertLess(
            float(linked_result.diagnostic("evidence_weight")),
            1.0,
        )
        self.assertEqual(
            linked_state.claims[0].source_event_ids,
            ("native-event",),
        )

    def test_native_transition_is_content_addressed_and_replayable(self) -> None:
        prior = PreferenceBelief.uniform()
        context, observation, provenance = event_fixture("balanced")
        updater = SemanticMemoryUpdater()
        view = make_update_view(
            updater.view_kind,
            context,
            observation,
            provenance,
            event_id="repeatable-event",
        )
        first = updater.update(updater.initial_state(prior), view)
        second = updater.update(updater.initial_state(prior), view)
        first_memory = first.state.opaque_state
        second_memory = second.state.opaque_state
        self.assertIsInstance(first_memory, NativeMemoryState)
        self.assertIsInstance(second_memory, NativeMemoryState)
        assert isinstance(first_memory, NativeMemoryState)
        assert isinstance(second_memory, NativeMemoryState)
        self.assertEqual(first_memory.state_id, second_memory.state_id)
        self.assertEqual(first.state.belief, second.state.belief)

    def test_native_updaters_reject_broader_or_narrower_views(self) -> None:
        prior = PreferenceBelief.uniform()
        context, observation, provenance = event_fixture()
        linked = ProvenanceLinkedMemoryUpdater()
        full_context = make_update_view(
            UpdateViewKind.FULL_CONTEXT,
            context,
            observation,
            provenance,
            event_id="wrong-view",
        )
        with self.assertRaises(ValueError):
            linked.update(linked.initial_state(prior), full_context)


class BlindedDecoderTests(unittest.TestCase):
    def test_two_decoder_views_are_distinct_and_blinded(self) -> None:
        prior = PreferenceBelief.uniform()
        context, observation, provenance = event_fixture()
        updater = ProvenanceLinkedMemoryUpdater()
        view = make_update_view(
            updater.view_kind,
            context,
            observation,
            provenance,
            event_id=(
                "user-7:incorrect:soft_profile_conditioned:"
                "provenance_linked_memory"
            ),
        )
        result = updater.update(updater.initial_state(prior), view)
        state = result.state.opaque_state
        self.assertIsInstance(state, NativeMemoryState)
        assert isinstance(state, NativeMemoryState)

        views = blinded_decoder_views(state)
        self.assertEqual(len(views), 2)
        self.assertNotEqual(views[0].decoder_id, views[1].decoder_id)
        self.assertNotEqual(views[0].payload_json, views[1].payload_json)
        self.assertEqual(
            views[0].pseudonymous_state_id,
            views[1].pseudonymous_state_id,
        )
        forbidden = {
            "system_id",
            "updater_id",
            "memory_kind",
            "latent_truth",
            "truth",
            "user_id",
        }
        for decoder_view in views:
            payload = decoder_view.payload()
            self.assertTrue(forbidden.isdisjoint(payload))
            serialized = json.dumps(payload).lower()
            for key in forbidden:
                self.assertNotIn(f'"{key}"', serialized)
            for hidden_label in (
                "user-7",
                "incorrect",
                "soft_profile_conditioned",
                "provenance_linked_memory",
            ):
                self.assertNotIn(hidden_label, serialized)

        decoded = decode_native_state(state)
        self.assertEqual(
            {item.decoder_id for item in decoded},
            {"direct_semantic_v1", "history_evidence_v1"},
        )
        for item in decoded:
            self.assertAlmostEqual(sum(item.belief.probabilities), 1.0)
        # Both fixed decoders are retained; callers cannot choose one silently.
        self.assertEqual(len(decoded), 2)


if __name__ == "__main__":
    unittest.main()
