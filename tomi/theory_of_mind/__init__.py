"""tomi.theory_of_mind — ToM datasets, tasks, and evaluation harness."""

from tomi.theory_of_mind.belief_tracking import (
    BeliefState,
    BeliefTrackingTask,
    build_belief_tracking_dataset,
    make_multi_step_task,
)
from tomi.theory_of_mind.evaluation import (
    EvaluationReport,
    TaskResult,
    ToMEvaluator,
)
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

__all__ = [
    "FalseBelief",
    "make_sally_anne",
    "make_maxi",
    "build_false_belief_dataset",
    "PerspectiveTakingTask",
    "make_visual_access_task",
    "make_verbal_comm_task",
    "build_perspective_taking_dataset",
    "BeliefState",
    "BeliefTrackingTask",
    "make_multi_step_task",
    "build_belief_tracking_dataset",
    "ToMEvaluator",
    "EvaluationReport",
    "TaskResult",
]
