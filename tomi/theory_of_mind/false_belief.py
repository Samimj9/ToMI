"""
tomi/theory_of_mind/false_belief.py
--------------------------------------
False Belief Task (FBT) dataset builder for Theory-of-Mind research.

The canonical false-belief task tests whether a model can attribute a
belief to an agent that differs from the true world state.

Classic example (Sally-Anne test)
-----------------------------------
    Sally places the marble in the basket.
    Sally leaves the room.
    Anne moves the marble to the box.
    Sally comes back.
    Where will Sally look for the marble?

    Correct (belief-consistent) answer : basket
    Incorrect (reality-consistent) answer: box

The model is expected to answer from *Sally's perspective* (basket), even
though the current world state has the marble in the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FalseBelief:
    """A single false-belief task instance.

    Attributes
    ----------
    prompt:
        The full text prompt presented to the model.
    belief_answer:
        The string the model should output if it correctly tracks the
        agent's (false) belief.
    reality_answer:
        The string corresponding to the true world state.
    agent:
        Name of the agent whose belief is being tracked.
    object_:
        The object being tracked.
    initial_location:
        Where the object starts (and where the agent believes it is).
    final_location:
        Where the object actually is after the move.
    metadata:
        Arbitrary extra fields (task variant, difficulty, …).
    """

    prompt: str
    belief_answer: str
    reality_answer: str
    agent: str
    object_: str
    initial_location: str
    final_location: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def is_false_belief(self) -> bool:
        """``True`` — this task always involves a false belief."""
        return True


# ---------------------------------------------------------------------------
# Built-in task templates
# ---------------------------------------------------------------------------

_SALLY_ANNE_TEMPLATE = (
    "{agent} places the {object_} in the {initial_location}.\n"
    "{agent} leaves the room.\n"
    "{mover} moves the {object_} to the {final_location}.\n"
    "{agent} returns.\n"
    "Where will {agent} look for the {object_}?"
)

_MAXI_TEMPLATE = (
    "{agent} puts the {object_} in the {initial_location}.\n"
    "{agent} goes to play outside.\n"
    "While {agent} is away, {mover} moves the {object_} to the {final_location}.\n"
    "{agent} comes back and wants the {object_}.\n"
    "Where will {agent} look?"
)


def make_sally_anne(
    agent: str = "Sally",
    mover: str = "Anne",
    object_: str = "marble",
    initial_location: str = "basket",
    final_location: str = "box",
    template: str = _SALLY_ANNE_TEMPLATE,
) -> FalseBelief:
    """Create a Sally-Anne style false-belief task.

    Parameters
    ----------
    agent:
        The agent whose false belief is tested.
    mover:
        The agent who moves the object.
    object_:
        The object being tracked.
    initial_location:
        Where the object starts.
    final_location:
        Where the object ends up.
    template:
        Prompt template string.

    Returns
    -------
    FalseBelief
    """
    prompt = template.format(
        agent=agent,
        mover=mover,
        object_=object_,
        initial_location=initial_location,
        final_location=final_location,
    )
    return FalseBelief(
        prompt=prompt,
        belief_answer=initial_location,
        reality_answer=final_location,
        agent=agent,
        object_=object_,
        initial_location=initial_location,
        final_location=final_location,
        metadata={"variant": "sally_anne", "mover": mover},
    )


def make_maxi(
    agent: str = "Maxi",
    mover: str = "his mother",
    object_: str = "chocolate",
    initial_location: str = "green cupboard",
    final_location: str = "blue cupboard",
) -> FalseBelief:
    """Create a Maxi-style false-belief task.

    Parameters
    ----------
    agent:
        The agent (Maxi, a child).
    mover:
        Who moves the chocolate while Maxi is away.
    object_:
        The object being tracked.
    initial_location:
        Starting location.
    final_location:
        Final location (unknown to the agent).

    Returns
    -------
    FalseBelief
    """
    prompt = _MAXI_TEMPLATE.format(
        agent=agent,
        mover=mover,
        object_=object_,
        initial_location=initial_location,
        final_location=final_location,
    )
    return FalseBelief(
        prompt=prompt,
        belief_answer=initial_location,
        reality_answer=final_location,
        agent=agent,
        object_=object_,
        initial_location=initial_location,
        final_location=final_location,
        metadata={"variant": "maxi"},
    )


# ---------------------------------------------------------------------------
# Batch dataset builder
# ---------------------------------------------------------------------------

def build_false_belief_dataset(
    n_variants: int = 10,
    include_sally_anne: bool = True,
    include_maxi: bool = True,
) -> List[FalseBelief]:
    """Build a small false-belief dataset with multiple variants.

    Parameters
    ----------
    n_variants:
        Number of (agent, location) variants to generate per template type.
    include_sally_anne:
        Include Sally-Anne variants.
    include_maxi:
        Include Maxi variants.

    Returns
    -------
    List[FalseBelief]
    """
    agents = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Iris", "Jack"]
    movers = ["Tom", "Lucy", "Mark", "Jane", "Peter", "Sara", "Chris", "Amy", "Leo", "Eva"]
    objects = ["marble", "ball", "key", "book", "toy", "coin", "ring", "pen", "cup", "stone"]
    locs_a = ["basket", "bag", "drawer", "box", "shelf", "pocket", "bin", "jar", "bowl", "chest"]
    locs_b = ["box", "locker", "cabinet", "sack", "crate", "pouch", "tray", "bucket", "case", "bin"]

    dataset: List[FalseBelief] = []

    if include_sally_anne:
        for i in range(min(n_variants, len(agents))):
            task = make_sally_anne(
                agent=agents[i],
                mover=movers[i],
                object_=objects[i],
                initial_location=locs_a[i],
                final_location=locs_b[i],
            )
            dataset.append(task)

    if include_maxi:
        for i in range(min(n_variants, len(agents))):
            task = make_maxi(
                agent=agents[i],
                mover=movers[i],
                object_=objects[i],
                initial_location=locs_a[i],
                final_location=locs_b[i],
            )
            dataset.append(task)

    return dataset
