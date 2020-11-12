import typing
from abc import ABC, abstractmethod

from loreleai.reasoning.lp import LPSolver
from loreleai.learning.task import Knowledge, Task
from loreleai.language.commons import Clause,Atom,Procedure
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.eval_functions import EvalFunction
import datetime



"""
This is an abstract learner class that defines a learner with the configurable options.

It follows a very simple learning principle: it iteratively
                                                     - searches for a single clause that covers most positive examples
                                                     - adds it to the program
                                                     - removes covered examples
                                                     
It is implemented as a template learner - you still need to provide the following methods:
                                                    - initialisation of the candidate pool (the data structure that keeps all candidates)
                                                    - getting a single candidate from the candidate pool
                                                    - adding candidate(s) to the pool
                                                    - evaluating a single candidate
                                                    - stopping the search for a single clause
                                                    - processing expansion/refinements of clauses
                                                    
The learner does not handle recursions correctly!
"""
class TemplateLearner(ABC):

    def __init__(self, solver_instance: LPSolver, eval_fn: EvalFunction, do_print=False):
        self._solver = solver_instance
        self._candidate_pool = []
        self._eval_fn = eval_fn
        self._print = do_print

    def _assert_knowledge(self, knowledge: Knowledge):
        """
        Assert knowledge into Prolog engine
        """
        facts = knowledge.get_atoms()
        for f_ind in range(len(facts)):
            self._solver.assert_fact(facts[f_ind])
            # self._solver.assertz(facts[f_ind])

        clauses = knowledge.get_clauses()
        for cl_ind in range(len(clauses)):
            self._solver.assert_rule(clauses[cl_ind])
            # self._solver.assertz(clauses[cl_ind])

    def _execute_program(self, clause: Clause) -> typing.Sequence[Atom]:
        """
        Evaluates a clause using the Prolog engine and background knowledge

        Returns a set of atoms that the clause covers
        """
        if len(clause.get_body().get_literals()) == 0:
            return []
        else:
            head_predicate = clause.get_head().get_predicate()
            head_variables = clause.get_head_variables()

            sols = self._solver.query(*clause.get_body().get_literals())

            sols = [head_predicate(*[s[v] for v in head_variables]) for s in sols]

            return sols

    @abstractmethod
    def initialise_pool(self):
        """
        Creates an empty pool of candidates
        """
        raise NotImplementedError()

    @abstractmethod
    def get_from_pool(self) -> Clause:
        """
        Gets a single clause from the pool
        """
        raise NotImplementedError()

    @abstractmethod
    def put_into_pool(self, candidates: typing.Union[Clause, Procedure, typing.Sequence]) -> None:
        """
        Inserts a clause/a set of clauses into the pool
        """
        raise NotImplementedError()

    def evaluate(self, examples: Task, clause: Clause) -> typing.Union[int, float]:
        """
        Evaluates a clause evaluating the Learner's eval_fn
        Returns a number (the higher the better)
        """
        covered = self._execute_program(clause)
        return self._eval_fn.evaluate(clause,examples,covered)

    @abstractmethod
    def stop_inner_search(self, eval: typing.Union[int, float], examples: Task, clause: Clause) -> bool:
        """
        Returns true if the search for a single clause should be stopped
        """
        raise NotImplementedError()

    @abstractmethod
    def process_expansions(self, examples: Task, exps: typing.Sequence[Clause], hypothesis_space: TopDownHypothesisSpace) -> typing.Sequence[Clause]:
        """
        Processes the expansions of a clause
        It can be used to eliminate useless expansions (e.g., the one that have no solution, ...)

        Returns a filtered set of candidates
        """
        raise NotImplementedError()

    def _learn_one_clause(self, examples: Task, hypothesis_space: TopDownHypothesisSpace) -> Clause:
        """
        Learns a single clause

        Returns a clause
        """
        # reset the search space
        hypothesis_space.reset_pointer()

        # empty the pool just in case
        self.initialise_pool()

        # put initial candidates into the pool
        self.put_into_pool(hypothesis_space.get_current_candidate())
        current_cand = None
        score = -100

        while current_cand is None or (len(self._candidate_pool) > 0 and not self.stop_inner_search(score, examples, current_cand)):
            # get first candidate from the pool
            current_cand = self.get_from_pool()

            # expand the candidate
            _ = hypothesis_space.expand(current_cand)
            # this is important: .expand() method returns candidates only the first time it is called;
            #     if the same node is expanded the second time, it returns the empty list
            #     it is safer than to use the .get_successors_of method
            exps = hypothesis_space.get_successors_of(current_cand)
            exps = self.process_expansions(examples, exps, hypothesis_space)
            # add into pool
            self.put_into_pool(exps)

            score = self.evaluate(examples, current_cand)

        if self._print:
            print(f"- New clause: {current_cand}")
            print(f"- Candidates has value {round(score,2)} for metric '{self._eval_fn.name()}'")
        return current_cand
        

    def learn(self, examples: Task, knowledge: Knowledge, hypothesis_space: TopDownHypothesisSpace):
        """
        General learning loop
        """

        self._assert_knowledge(knowledge)
        final_program = []
        examples_to_use = examples
        pos, _ = examples_to_use.get_examples()
        i = 0
        start = datetime.datetime.now()


        while len(final_program) == 0 or len(pos) > 0:
            # learn na single clause

            if self._print:
                print(f"Iteration {i}")
                print("- Current program:")
                for program_clause in final_program:
                    print("\t"+str(program_clause))
                
            cl = self._learn_one_clause(examples_to_use, hypothesis_space)
            final_program.append(cl)

            # update covered positive examples
            covered = self._execute_program(cl)

            pos, neg = examples_to_use.get_examples()
            pos = pos.difference(covered)

            examples_to_use = Task(pos, neg)
            i += 1

        if self._print:
            print("Done! Search took {:.5f} seconds.".format((datetime.datetime.now()-start).total_seconds()))

        return final_program
