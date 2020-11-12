import typing
from abc import ABC, abstractmethod

from loreleai.learning.abstract_learners import TemplateLearner
from loreleai.reasoning.lp import LPSolver
from loreleai.language.commons import Clause,Atom,Procedure
from loreleai.learning.task import Task, Knowledge
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.eval_functions import EvalFunction

from orderedset import OrderedSet



"""
A simple breadth-first top-down learner: it extends the template learning by searching in a breadth-first fashion

It implements the abstract functions in the following way:
  - initialise_pool: creates an empty OrderedSet
  - put_into_pool: adds to the ordered set
  - get_from_pool: returns the first elements in the ordered set
  - evaluate: returns the number of covered positive examples and 0 if any negative example is covered
  - stop inner search: stops if the provided score of a clause is bigger than zero 
  - process expansions: removes from the hypothesis space all clauses that have no solutions

The learner does not handle recursions correctly!
"""
class SimpleBreadthFirstLearner(TemplateLearner):

    def __init__(self, solver_instance: LPSolver, eval_fn: EvalFunction, max_body_literals=4,do_print=False):
        super().__init__(solver_instance,eval_fn,do_print=do_print)
        self._max_body_literals = max_body_literals

    def initialise_pool(self):
        self._candidate_pool = OrderedSet()

    def put_into_pool(self, candidates: typing.Union[Clause, Procedure, typing.Sequence]) -> None:
        if isinstance(candidates, Clause):
            self._candidate_pool.add(candidates)
        else:
            self._candidate_pool |= candidates

    def get_from_pool(self) -> Clause:
        return self._candidate_pool.pop(0)


    def stop_inner_search(self, eval: typing.Union[int, float], examples: Task, clause: Clause) -> bool:
        if eval > 0:
            return True
        else:
            return False

    def process_expansions(self, examples: Task, exps: typing.Sequence[Clause], hypothesis_space: TopDownHypothesisSpace) -> typing.Sequence[Clause]:
        # eliminate every clause with more body literals than allowed
        exps = [cl for cl in exps if len(cl) <= self._max_body_literals]

        # check if every clause has solutions
        exps = [(cl, self._solver.has_solution(*cl.get_body().get_literals())) for cl in exps]
        new_exps = []

        for ind in range(len(exps)):
            if exps[ind][1]:
                # keep it if it has solutions
                new_exps.append(exps[ind][0])
            else:
                # remove from hypothesis space if it does not
                hypothesis_space.remove(exps[ind][0])

        return new_exps