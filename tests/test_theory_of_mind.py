"""
tests/test_theory_of_mind.py
------------------------------
Unit tests for Theory-of-Mind task builders.
"""

from __future__ import annotations

import pytest

from tomi.theory_of_mind.false_belief import (
    FalseBelief,
    build_false_belief_dataset,
    make_maxi,
    make_sally_anne,
)
from tomi.theory_of_mind.perspective_taking import (
    PerspectiveTakingTask,
    build_perspective_taking_dataset,
    make_verbal_comm_task,
    make_visual_access_task,
)
from tomi.theory_of_mind.belief_tracking import (
    BeliefState,
    BeliefTrackingTask,
    build_belief_tracking_dataset,
    make_multi_step_task,
)


# ---------------------------------------------------------------------------
# False Belief
# ---------------------------------------------------------------------------

class TestFalseBelief:
    def test_make_sally_anne_defaults(self):
        task = make_sally_anne()
        assert isinstance(task, FalseBelief)
        assert "Sally" in task.prompt
        assert task.belief_answer == "basket"
        assert task.reality_answer == "box"
        assert task.is_false_belief

    def test_make_sally_anne_custom(self):
        task = make_sally_anne(agent="Alice", initial_location="cupboard", final_location="drawer")
        assert "Alice" in task.prompt
        assert task.belief_answer == "cupboard"
        assert task.reality_answer == "drawer"

    def test_make_maxi(self):
        task = make_maxi()
        assert isinstance(task, FalseBelief)
        assert "Maxi" in task.prompt
        assert task.belief_answer == "green cupboard"

    def test_build_dataset_count(self):
        dataset = build_false_belief_dataset(n_variants=5)
        assert len(dataset) == 10  # 5 sally-anne + 5 maxi

    def test_build_dataset_only_sally_anne(self):
        dataset = build_false_belief_dataset(n_variants=3, include_maxi=False)
        assert len(dataset) == 3

    def test_dataset_types(self):
        dataset = build_false_belief_dataset(n_variants=2)
        assert all(isinstance(t, FalseBelief) for t in dataset)


# ---------------------------------------------------------------------------
# Perspective Taking
# ---------------------------------------------------------------------------

class TestPerspectiveTaking:
    def test_visual_access_task(self):
        task = make_visual_access_task()
        assert isinstance(task, PerspectiveTakingTask)
        assert "John" in task.prompt
        assert task.agent == "John"
        assert not task.has_access

    def test_verbal_comm_task(self):
        task = make_verbal_comm_task()
        assert isinstance(task, PerspectiveTakingTask)
        assert not task.has_access

    def test_build_dataset(self):
        dataset = build_perspective_taking_dataset(n=5)
        # 5 visual + 5 verbal
        assert len(dataset) == 10
        assert all(isinstance(t, PerspectiveTakingTask) for t in dataset)


# ---------------------------------------------------------------------------
# Belief Tracking
# ---------------------------------------------------------------------------

class TestBeliefTracking:
    def test_default_task(self):
        task = make_multi_step_task()
        assert isinstance(task, BeliefTrackingTask)
        assert len(task.belief_states) == 3  # initial + 2 moves

    def test_belief_states_track_correctly(self):
        task = make_multi_step_task(
            agent="Tom",
            locations=["A", "B", "C"],
            agent_witness=[True, False],  # sees first move, misses second
        )
        # After step 1 (witnessed): believes B
        assert task.belief_states[1].believed_location == "B"
        # After step 2 (not witnessed): still believes B
        assert task.belief_states[2].believed_location == "B"
        assert task.belief_states[2].actual_location == "C"
        assert task.belief_states[2].is_false

    def test_expected_answer_is_believed_location(self):
        task = make_multi_step_task(
            locations=["X", "Y", "Z"],
            agent_witness=[False, False],
        )
        # Never witnesses a move; still believes X
        assert task.expected_answer == "X"
        assert task.final_belief_state.believed_location == "X"

    def test_build_dataset(self):
        dataset = build_belief_tracking_dataset(n=4)
        assert len(dataset) == 4
        assert all(isinstance(t, BeliefTrackingTask) for t in dataset)

    def test_witness_all_moves(self):
        task = make_multi_step_task(
            locations=["A", "B", "C"],
            agent_witness=[True, True],
        )
        # Witnesses all: believes current location
        assert task.expected_answer == "C"
        assert not task.final_belief_state.is_false

    def test_invalid_witness_length_raises(self):
        with pytest.raises(AssertionError):
            make_multi_step_task(
                locations=["A", "B", "C"],
                agent_witness=[True],  # Should be len 2
            )
