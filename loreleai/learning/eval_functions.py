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

    def __init__(self,name):
        self._name = name

    @abstractmethod
    def evaluate(self, clause: Clause, examples: Task,covered: Sequence[Atom]):
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
    def __init__(self):
        super().__init__("Accuracy")

    def evaluate(self, clause: Clause, examples: Task, covered: Sequence[Atom]):
        pos,neg = examples.get_examples()
        covered_pos = len(pos.intersection(covered))
        covered_neg = len(neg.intersection(covered))

        if covered_pos + covered_neg == 0:
            return 0
        return covered_pos / (covered_pos + covered_neg)

class Compression(EvalFunction):
    """
    Compression is similar to coverage but favours shorter clauses
    """
    def __init__(self):
        super().__init__("Compression")

    def evaluate(self, clause: Clause, examples: Task, covered: Sequence[Atom]):
        pos,neg = examples.get_examples()
        covered_pos = len(pos.intersection(covered))
        covered_neg = len(neg.intersection(covered))
        clause_length = len(clause.get_literals())
        return covered_pos - covered_neg - clause_length + 1

class Coverage(EvalFunction):
    """
    Coverage is defined as the difference between the number of positive
    and negative examples covered
    """
    def __init__(self):
        super().__init__("Coverage")

    def evaluate(self, clause: Clause, examples: Task, covered: Sequence[Atom]):
        pos,neg = examples.get_examples()
        covered_pos = len(pos.intersection(covered))
        covered_neg = len(neg.intersection(covered))
        return covered_pos - covered_neg

class Entropy(EvalFunction):
    """
    Entropy is a measure of how well the clause divides 
    negative and positive examples into two distinct categories.
    This implementation uses:

    p * log10(p) + (1-p) * log10(1-p), with p = P/(P+N). 
    P and N are respectively
    the number of positive and negative examples that are covered by the clause
    """
    def __init__(self):
        super().__init__("Entropy")

    def evaluate(self, clause: Clause, examples: Task, covered: Sequence[Atom]):
        pos,neg = examples.get_examples()
        covered_pos = len(pos.intersection(covered))
        covered_neg = len(neg.intersection(covered)) 
        if covered_pos+covered_neg == 0:
            return 0

        p = covered_pos/(covered_pos+covered_neg)
        return p*math.log10(p) + (1-p)*math.log10(1-p)    
    