"""
tomi/theory_of_mind/belief_tracking.py
-----------------------------------------
Multi-step belief-tracking task dataset.

Belief tracking probes the model's ability to maintain and update an
agent's belief state across several events, some of which the agent
witnesses and some of which they do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BeliefState:
    """Snapshot of an agent's belief about the world.

    Attributes
    ----------
    agent:
        The agent.
    believed_location:
        Where the agent believes the object to be.
    actual_location:
        Where the object actually is.
    step:
        Which event step this state reflects.
    """

    agent: str
    believed_location: str
    actual_location: str
    step: int

    @property
    def is_false(self) -> bool:
        """``True`` if the agent has a false belief."""
        return self.believed_location != self.actual_location


@dataclass
class BeliefTrackingTask:
    """A multi-step belief tracking task.

    Attributes
    ----------
    prompt:
        The full prompt with all events.
    agent:
        The agent whose beliefs are tracked.
    object_:
        The tracked object.
    belief_states:
        Ordered list of belief state snapshots after each event.
    query_step:
        Which step the question asks about.
    expected_answer:
        Expected model answer (believed location at query_step).
    metadata:
        Extra fields.
    """

    prompt: str
    agent: str
    object_: str
    belief_states: List[BeliefState]
    query_step: int
    expected_answer: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def final_belief_state(self) -> BeliefState:
        """The belief state at ``query_step``."""
        return self.belief_states[self.query_step]


def make_multi_step_task(
    agent: str = "Tom",
    mover: str = "Lisa",
    object_: str = "keys",
    locations: Optional[List[str]] = None,
    agent_witness: Optional[List[bool]] = None,
) -> BeliefTrackingTask:
    """Create a multi-step belief tracking task.

    The task simulates an object being moved across several locations while
    the agent may or may not witness each move.

    Parameters
    ----------
    agent:
        The agent whose belief is tracked.
    mover:
        The other agent who moves the object.
    object_:
        The tracked object.
    locations:
        Ordered list of locations the object passes through (≥ 3).
    agent_witness:
        Boolean list indicating whether the agent witnesses each move.
        Must have ``len(locations) - 1`` elements.

    Returns
    -------
    BeliefTrackingTask
    """
    if locations is None:
        locations = ["kitchen", "living room", "bedroom"]
    if agent_witness is None:
        agent_witness = [True, False]  # one move witnessed, one not

    assert len(agent_witness) == len(locations) - 1, (
        "agent_witness must have len(locations)-1 entries"
    )

    events: List[str] = [
        f"{agent} puts the {object_} in the {locations[0]}."
    ]
    belief_states: List[BeliefState] = [
        BeliefState(
            agent=agent,
            believed_location=locations[0],
            actual_location=locations[0],
            step=0,
        )
    ]

    believed_loc = locations[0]

    for step, (src, dst, witnessed) in enumerate(
        zip(locations[:-1], locations[1:], agent_witness), start=1
    ):
        if witnessed:
            events.append(
                f"{mover} moves the {object_} from the {src} to the {dst}. "
                f"{agent} sees this."
            )
            believed_loc = dst
        else:
            events.append(
                f"While {agent} is away, {mover} moves the {object_} "
                f"from the {src} to the {dst}."
            )
            # believed_loc stays the same
        belief_states.append(
            BeliefState(
                agent=agent,
                believed_location=believed_loc,
                actual_location=dst,
                step=step,
            )
        )

    # Query at the final step
    query_step = len(belief_states) - 1
    events.append(
        f"Where does {agent} think the {object_} is?"
    )

    prompt = "\n".join(events)
    expected = belief_states[query_step].believed_location

    return BeliefTrackingTask(
        prompt=prompt,
        agent=agent,
        object_=object_,
        belief_states=belief_states,
        query_step=query_step,
        expected_answer=expected,
        metadata={"n_steps": str(len(locations) - 1)},
    )


def build_belief_tracking_dataset(
    n: int = 10,
    n_steps: int = 3,
) -> List[BeliefTrackingTask]:
    """Build a batch of belief tracking tasks.

    Parameters
    ----------
    n:
        Number of tasks.
    n_steps:
        Number of location moves per task.

    Returns
    -------
    List[BeliefTrackingTask]
    """
    agents   = ["Tom", "Anna", "Sam", "Kate", "Leo", "Mia", "Ben", "Zoe", "Dan", "Fay"]
    movers   = ["Lisa", "Mark", "Nina", "Jake", "Amy", "Rob", "Sue", "Ian", "Kay", "Eve"]
    objects  = ["keys", "book", "toy", "phone", "bag", "cup", "pen", "coin", "ring", "card"]
    base_locs = [
        ["kitchen", "living room", "bedroom", "bathroom"],
        ["garage", "garden", "shed", "attic"],
        ["office", "lobby", "lab", "cafeteria"],
        ["hall", "study", "closet", "basement"],
        ["shop", "storage", "stockroom", "checkout"],
    ]

    dataset: List[BeliefTrackingTask] = []
    for i in range(min(n, len(agents))):
        locs = base_locs[i % len(base_locs)][: n_steps + 1]
        # Alternate witnessing: odd moves are witnessed, even are not
        witnessed = [j % 2 == 0 for j in range(n_steps)]
        task = make_multi_step_task(
            agent=agents[i],
            mover=movers[i],
            object_=objects[i],
            locations=locs,
            agent_witness=witnessed,
        )
        dataset.append(task)

    return dataset
