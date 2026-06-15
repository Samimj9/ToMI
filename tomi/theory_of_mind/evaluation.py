"""
tomi/theory_of_mind/evaluation.py
------------------------------------
Evaluation harness for Theory-of-Mind benchmarks.

Runs a model on a ToM dataset and collects:

* Token-level logit differences between belief and reality answers.
* Accuracy (does the model's top prediction match the expected answer?).
* Activation caches for further analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch

from tomi.metrics.belief_metrics import BeliefScoreResult, belief_score
from tomi.models.base_model import ToMModel
from tomi.theory_of_mind.belief_tracking import BeliefTrackingTask
from tomi.theory_of_mind.false_belief import FalseBelief
from tomi.theory_of_mind.perspective_taking import PerspectiveTakingTask
from tomi.utils.logging import get_logger
from tomi.utils.tokenizer import get_token_ids

log = get_logger(__name__)

ToMTask = Union[FalseBelief, PerspectiveTakingTask, BeliefTrackingTask]


@dataclass
class TaskResult:
    """Result for a single ToM task.

    Attributes
    ----------
    task_index:
        Index of the task in the dataset.
    prompt:
        The prompt that was evaluated.
    belief_score_result:
        Detailed belief score.
    expected_answer:
        The answer consistent with the agent's belief.
    predicted_answer:
        The model's actual top prediction.
    is_correct:
        Whether the model correctly answered from the belief perspective.
    """

    task_index: int
    prompt: str
    belief_score_result: BeliefScoreResult
    expected_answer: str
    predicted_answer: str
    is_correct: bool


@dataclass
class EvaluationReport:
    """Aggregated results across a ToM dataset.

    Attributes
    ----------
    task_results:
        Individual :class:`TaskResult` objects.
    accuracy:
        Fraction of tasks answered correctly (from the belief perspective).
    mean_belief_logit_diff:
        Average logit difference ``logit(belief) - logit(reality)``.
    mean_belief_prob:
        Average probability on the belief answer.
    n_tasks:
        Total number of evaluated tasks.
    """

    task_results: List[TaskResult] = field(default_factory=list)
    accuracy: float = 0.0
    mean_belief_logit_diff: float = 0.0
    mean_belief_prob: float = 0.0
    n_tasks: int = 0

    def __post_init__(self) -> None:
        if self.task_results and self.n_tasks == 0:
            self._compute_aggregates()

    def _compute_aggregates(self) -> None:
        """Recompute aggregate statistics from individual results."""
        n = len(self.task_results)
        if n == 0:
            return
        self.n_tasks = n
        self.accuracy = sum(r.is_correct for r in self.task_results) / n
        self.mean_belief_logit_diff = (
            sum(r.belief_score_result.belief_logit_diff for r in self.task_results) / n
        )
        self.mean_belief_prob = (
            sum(r.belief_score_result.belief_prob for r in self.task_results) / n
        )

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"EvaluationReport\n"
            f"  Tasks          : {self.n_tasks}\n"
            f"  Accuracy       : {self.accuracy:.2%}\n"
            f"  Belief Logit Δ : {self.mean_belief_logit_diff:+.4f}\n"
            f"  Belief P(ans)  : {self.mean_belief_prob:.4f}\n"
        )


class ToMEvaluator:
    """Runs a ToMModel on Theory-of-Mind benchmarks.

    Parameters
    ----------
    model:
        The wrapped model to evaluate.
    position:
        Sequence position for metric evaluation (default ``-1``).
    """

    def __init__(self, model: ToMModel, position: int = -1) -> None:
        self.model = model
        self.position = position

    def evaluate(
        self,
        dataset: List[ToMTask],
        add_prefix: bool = False,
        prefix: str = "The answer is:",
    ) -> EvaluationReport:
        """Evaluate the model on a list of ToM tasks.

        Parameters
        ----------
        dataset:
            List of task instances (any supported ToM task type).
        add_prefix:
            If ``True``, appends *prefix* to each prompt to guide generation.
        prefix:
            String to append to prompts when *add_prefix* is ``True``.

        Returns
        -------
        EvaluationReport
        """
        results: List[TaskResult] = []

        for i, task in enumerate(dataset):
            result = self._evaluate_single(i, task, add_prefix=add_prefix, prefix=prefix)
            if result is not None:
                results.append(result)

        report = EvaluationReport(task_results=results)
        report._compute_aggregates()
        log.info(
            "Evaluation complete: %d/%d tasks, accuracy=%.2f%%",
            len(results),
            len(dataset),
            report.accuracy * 100,
        )
        return report

    def _evaluate_single(
        self,
        index: int,
        task: ToMTask,
        add_prefix: bool = False,
        prefix: str = "",
    ) -> Optional[TaskResult]:
        """Evaluate a single task.

        Returns ``None`` if tokenisation fails (e.g. multi-token answer).
        """
        prompt = task.prompt
        if add_prefix:
            prompt = prompt + "\n" + prefix

        # Determine belief / reality answers
        belief_answer, reality_answer = self._get_answers(task)
        if not belief_answer or not reality_answer:
            log.warning("Task %d: could not determine answers.", index)
            return None

        # Tokenise prompt
        tokens = self.model.tokenize(prompt, padding=False)
        input_ids = tokens["input_ids"]

        # Get answer token ids (first token of each answer string)
        try:
            belief_ids = get_token_ids(
                self.model.tokenizer, belief_answer, add_prefix_space=True
            )
            reality_ids = get_token_ids(
                self.model.tokenizer, reality_answer, add_prefix_space=True
            )
        except ValueError as exc:
            log.warning("Task %d: token lookup failed: %s", index, exc)
            return None

        belief_token_id = belief_ids[0]
        reality_token_id = reality_ids[0]

        # Forward pass
        logits = self.model.get_logits(input_ids)

        # Compute belief score
        bs = belief_score(logits, belief_token_id, reality_token_id, position=self.position)

        # Top prediction
        top_id = int(logits[0, self.position, :].argmax().item())
        predicted_answer = self.model.tokenizer.decode([top_id]).strip()

        return TaskResult(
            task_index=index,
            prompt=prompt,
            belief_score_result=bs,
            expected_answer=belief_answer,
            predicted_answer=predicted_answer,
            is_correct=bs.is_correct,
        )

    def _get_answers(self, task: ToMTask) -> Tuple[str, str]:
        """Extract belief and reality answers from any ToM task type."""
        if isinstance(task, FalseBelief):
            return task.belief_answer, task.reality_answer
        if isinstance(task, PerspectiveTakingTask):
            # The "correct" answer is what the agent knows; the foil is what they don't
            return task.knows_answer.split()[0], task.does_not_know_answer.split()[0]
        if isinstance(task, BeliefTrackingTask):
            final = task.final_belief_state
            return final.believed_location, final.actual_location
        return "", ""
