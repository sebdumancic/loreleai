import typing
from abc import ABC, abstractmethod

from loreleai.reasoning.lp import LPSolver
from loreleai.learning.task import Knowledge, Task
from loreleai.language.lp import Clause,Atom,Procedure
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace, HypothesisSpace
from loreleai.learning.eval_functions import EvalFunction
import datetime

class Learner(ABC):
    """
    Base class for all learners
    """
    def __init__(self):
        self._learnresult = LearnResult()

    @abstractmethod
    def learn(self, examples: Task, knowledge: Knowledge, hypothesis_space: HypothesisSpace):
        raise NotImplementedError()

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

class LearnResult:
    """
    The LearnResult class holds statistics about the learning process.
    It is implemented as a dict and supports indexing [] as usual.
    """
    def __init__(self):
        self.info = dict()

    def __getitem__(self,key):
        return self.info[key]

    def __setitem__(self,key,val):
        self.info[key] = val
    
    def __repr__(self):
        max_keylength = max([len(str(k)) for k in self.info.keys()])
        output_str = "Result of learning: \n"

        for key,val in self.info.items():
            output_str += str(key) + ":" + " "*(max_keylength - len(str(key))+2) + str(val) + "\n"
        return output_str



class TemplateLearner(Learner):

    def __init__(self, solver_instance: LPSolver, eval_fn: EvalFunction, do_print=False):
        self._solver = solver_instance
        self._candidate_pool = []
        self._eval_fn = eval_fn
        self._print = do_print
        
        # Statistics about learning process
        self._prolog_queries = 0
        self._intermediate_coverage = []     # Coverage of examples after every iteration

        super().__init__()

    def _assert_knowledge(self, knowledge: Knowledge):
        """
        Assert knowledge into Prolog engine
        """
        facts = knowledge.get_atoms()
        for f_ind in range(len(facts)):
            # self._solver.assert_fact(facts[f_ind])
            self._solver.assertz(facts[f_ind])

        clauses = knowledge.get_clauses()
        for cl_ind in range(len(clauses)):
            # self._solver.assert_rule(clauses[cl_ind])
            self._solver.assertz(clauses[cl_ind])

    def _execute_program(self, clause: Clause, count_as_query=True) -> typing.Sequence[Atom]:
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

            self._prolog_queries += 1 if count_as_query else 0

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

    def evaluate(self, examples: Task, clause: Clause, hypothesis_space: HypothesisSpace) -> typing.Union[int, float]:
        """
        Evaluates a clause by calling the Learner's eval_fn.
        Returns a number (the higher the better)
        """
        # add_to_cache(node,key,val)
        # retrieve_from_cache(node,key) -> val or None
        # remove_from_cache(node,key) -> None
        
        # Cache holds sets of examples that were covered before
        covered = hypothesis_space.retrieve_from_cache(clause,"covered")

        # We have executed this clause before
        if covered is not None:
            # Note that _eval.fn.evaluate() will ignore clauses in `covered`
            # that are not in the current Task
            result = self._eval_fn.evaluate(clause,examples,covered)
            # print("No query here.")
            return result
        else:
            covered = self._execute_program(clause)
            # if 'None', i.e. trivial hypothesis, all clauses are covered
            if covered is None:
                pos,neg = examples.get_examples()
                covered = pos.union(neg)

            result = self._eval_fn.evaluate(clause,examples,covered)
            hypothesis_space.add_to_cache(clause,"covered",covered)
            return result

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

            score = self.evaluate(examples, current_cand, hypothesis_space)

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

            # Find intermediate quality of program at this point, add to learnresult (don't cound these as Prolog queries)
            c = set()
            for cl in final_program:
                c = c.union(self._execute_program(cl,count_as_query=False))
            pos_covered = len(c.intersection(examples._positive_examples))
            neg_covered = len(c.intersection(examples._negative_examples))
            self.__intermediate_coverage.append((pos_covered,neg_covered))

            # Remove covered examples and start next iteration
            pos, neg = examples_to_use.get_examples()
            pos = pos.difference(covered)

            examples_to_use = Task(pos, neg)
            i += 1

        total_time = (datetime.datetime.now()-start).total_seconds()
        if self._print:
            print("Done! Search took {:.5f} seconds.".format(total_time))

        # Wrap results into learnresult and return
        self._learnresult["final_program"] = final_program
        self._learnresult["total_time"] = total_time
        self._learnresult["num_iterations"] = i
        self._learnresult["evalfn_evaluations"] = self._eval_fn._clauses_evaluated
        self._learnresult["prolog_queries"] = self._prolog_queries
        self._learnresult["intermediate_coverage"] = self._intermediate_coverage

        return self._learnresult
