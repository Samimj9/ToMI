"""
tomi/theory_of_mind/perspective_taking.py
------------------------------------------
Perspective-taking task dataset for Theory-of-Mind research.

Perspective-taking tasks probe whether a model can reason about what an
agent *knows* or *believes* based on their epistemic access to information
(what they have and haven't seen).

Example
-------
::

    John is in the garden. He cannot see inside the house.
    Mary enters the house and finds a letter on the table.
    What does John know about the letter?

    Expected: John does not know about the letter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PerspectiveTakingTask:
    """A perspective-taking task instance.

    Attributes
    ----------
    prompt:
        Full prompt text.
    agent:
        The agent whose perspective is probed.
    knows_answer:
        What the agent knows (expected model output when correctly tracking
        the agent's epistemic state).
    does_not_know_answer:
        What the agent does NOT know.
    observer:
        Another agent who has different epistemic access.
    information:
        The piece of information in question.
    has_access:
        Whether the primary agent has direct access to the information.
    metadata:
        Extra fields.
    """

    prompt: str
    agent: str
    knows_answer: str
    does_not_know_answer: str
    observer: str
    information: str
    has_access: bool
    metadata: Dict[str, str] = field(default_factory=dict)


_VISUAL_ACCESS_TEMPLATE = (
    "{agent} is in the {agent_location}. {agent} cannot see inside the {other_location}.\n"
    "{observer} goes into the {other_location} and finds a {information}.\n"
    "What does {agent} know about the {information}?"
)

_VERBAL_COMM_TEMPLATE = (
    "{observer} told {agent} about the {information}.\n"
    "{agent} then left for the {agent_location}.\n"
    "Later, {observer} discovered new information about the {information} "
    "but could not reach {agent}.\n"
    "What does {agent} know about the {information}?"
)


def make_visual_access_task(
    agent: str = "John",
    observer: str = "Mary",
    agent_location: str = "garden",
    other_location: str = "house",
    information: str = "letter",
) -> PerspectiveTakingTask:
    """Create a visual-access perspective-taking task.

    The agent lacks visual access to a location where information appears.

    Parameters
    ----------
    agent:
        Agent who lacks access.
    observer:
        Agent who has access.
    agent_location:
        Where the agent is.
    other_location:
        The location the agent cannot see.
    information:
        The information item.

    Returns
    -------
    PerspectiveTakingTask
    """
    prompt = _VISUAL_ACCESS_TEMPLATE.format(
        agent=agent,
        observer=observer,
        agent_location=agent_location,
        other_location=other_location,
        information=information,
    )
    return PerspectiveTakingTask(
        prompt=prompt,
        agent=agent,
        knows_answer=f"{agent} does not know about the {information}.",
        does_not_know_answer=f"{agent} knows about the {information}.",
        observer=observer,
        information=information,
        has_access=False,
        metadata={"variant": "visual_access"},
    )


def make_verbal_comm_task(
    agent: str = "Alice",
    observer: str = "Bob",
    agent_location: str = "market",
    information: str = "package",
) -> PerspectiveTakingTask:
    """Create a verbal-communication perspective-taking task.

    The agent was told information but then left; the observer learned an
    update but couldn't communicate it.

    Parameters
    ----------
    agent:
        Agent who received initial but not updated information.
    observer:
        Agent who knows the update.
    agent_location:
        Where the agent went.
    information:
        The information item.

    Returns
    -------
    PerspectiveTakingTask
    """
    prompt = _VERBAL_COMM_TEMPLATE.format(
        agent=agent,
        observer=observer,
        agent_location=agent_location,
        information=information,
    )
    return PerspectiveTakingTask(
        prompt=prompt,
        agent=agent,
        knows_answer=f"{agent} knows only the original information about the {information}.",
        does_not_know_answer=f"{agent} knows the updated information about the {information}.",
        observer=observer,
        information=information,
        has_access=False,
        metadata={"variant": "verbal_comm"},
    )


def build_perspective_taking_dataset(n: int = 10) -> List[PerspectiveTakingTask]:
    """Build a batch of perspective-taking tasks.

    Parameters
    ----------
    n:
        Number of tasks per variant.

    Returns
    -------
    List[PerspectiveTakingTask]
    """
    agents   = ["John", "Alice", "Mark", "Emma", "David", "Sara", "Paul", "Lisa", "Tom", "Anna"]
    observers = ["Mary", "Bob", "Lucy", "Frank", "Grace", "Leo", "Amy", "Mike", "Eva", "Jack"]
    locations = ["garden", "park", "office", "market", "library", "cafe", "gym", "beach", "lab", "studio"]
    rooms    = ["house", "room", "building", "hall", "store", "warehouse", "cellar", "attic", "shed", "vault"]
    items    = ["letter", "package", "note", "message", "parcel", "document", "report", "file", "key", "card"]

    dataset: List[PerspectiveTakingTask] = []
    for i in range(min(n, len(agents))):
        dataset.append(
            make_visual_access_task(
                agent=agents[i],
                observer=observers[i],
                agent_location=locations[i],
                other_location=rooms[i],
                information=items[i],
            )
        )
    for i in range(min(n, len(agents))):
        dataset.append(
            make_verbal_comm_task(
                agent=agents[i],
                observer=observers[i],
                agent_location=locations[i],
                information=items[i],
            )
        )
    return dataset
