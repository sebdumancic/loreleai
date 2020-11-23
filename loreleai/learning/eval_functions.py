import typing
from abc import ABC, abstractmethod
from typing import Sequence
from loreleai.language.commons import Clause, Atom
from loreleai.learning.task import Task
import math


class EvalFunction(ABC):
    """
    Abstract base class for an evaluation function
    """

    def __init__(self, name):
        self._clauses_evaluated = 0
        self._name = name

    @abstractmethod
    def evaluate(self, clause: Clause, examples: Task, covered: Sequence[Atom]):
        """
        Evaluates the quality of clause on the given examples and
        set of covered atoms
        """
        raise NotImplementedError()

    def name(self):
        return self._name


class Accuracy(EvalFunction):
    """
    Accuracy is defined as the number of positive examples coverd,
    divided by the number of positive and negative examples covered
    """

    def __init__(self, return_upperbound=False):
        super().__init__("Accuracy")
        self._return_upperbound = return_upperbound

    def evaluate(self, clause: Clause, examples: Task, covered: Sequence[Atom]):
        self._clauses_evaluated += 1

        pos, neg = examples.get_examples()
        covered_pos = len(pos.intersection(covered))
        covered_neg = len(neg.intersection(covered))

        if covered_pos + covered_neg == 0:
            return 0 if not self._return_upperbound else 0, 0
        return (
            covered_pos / (covered_pos + covered_neg)
            if not self._return_upperbound
            else covered_pos / (covered_pos + covered_neg),
            1,
        )


class Compression(EvalFunction):
    """
    Compression is similar to coverage but favours shorter clauses
    """

    def __init__(self, return_upperbound=False):
        super().__init__("Compression")
        self._return_upperbound = return_upperbound

    def evaluate(self, clause: Clause, examples: Task, covered: Sequence[Atom]):
        self._clauses_evaluated += 1

        pos, neg = examples.get_examples()
        covered_pos = len(pos.intersection(covered))
        covered_neg = len(neg.intersection(covered))
        clause_length = len(clause.get_literals())
        if self._return_upperbound:
            return (covered_pos - covered_neg - clause_length + 1), covered_pos
        return covered_pos - covered_neg - clause_length + 1


class Coverage(EvalFunction):
    """
    Coverage is defined as the difference between the number of positive
    and negative examples covered.
    """

    def __init__(self, return_upperbound=False):
        """
        Initializes the Coverage EvalFunction. When return_upperbound is True,
        a tuple (coverage, upper_bound) will be returned upon evaluation, where upper_bound
        gives the maximum coverage any clauses extending the original clause can achieve
        """
        super().__init__("Coverage")
        self._return_upperbound = return_upperbound

    def evaluate(self, clause: Clause, examples: Task, covered: Sequence[Atom]):
        self._clauses_evaluated += 1

        pos, neg = examples.get_examples()
        covered_pos = len(pos.intersection(covered))
        covered_neg = len(neg.intersection(covered))
        if self._return_upperbound:
            return (covered_pos - covered_neg), covered_pos
        return covered_pos - covered_neg


class Entropy(EvalFunction):
    """
    Entropy is a measure of how well the clause divides
    negative and positive examples into two distinct categories.
    This implementation uses:

    -(p * log10(p) + (1-p) * log10(1-p)), with p = P/(P+N).
    P and N are respectively
    the number of positive and negative examples that are covered by the clause
    """

    def __init__(self):
        super().__init__("Entropy")

    def evaluate(self, clause: Clause, examples: Task, covered: Sequence[Atom]):
        self._clauses_evaluated += 1
    
        pos, neg = examples.get_examples()
        covered_pos = len(pos.intersection(covered))
        covered_neg = len(neg.intersection(covered))
        if covered_pos + covered_neg == 0:
            return 0

        p = covered_pos / (covered_pos + covered_neg)

        # Perfect split, no entropy
        if p == 1 or p == 0:
            return 0
        return -(p * math.log10(p) + (1 - p) * math.log10(1 - p))
