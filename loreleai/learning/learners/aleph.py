import typing
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple
from orderedset import OrderedSet
import datetime

from loreleai.learning.abstract_learners import Learner, TemplateLearner, LearnResult
from loreleai.learning import Task, Knowledge, HypothesisSpace, TopDownHypothesisSpace
from loreleai.language.lp import (
    Clause,
    Constant,
    c_type,
    Variable,
    Not,
    Atom,
    Procedure,
    Body
)
from itertools import product, combinations_with_replacement
from collections import Counter
from loreleai.reasoning.lp import LPSolver
from loreleai.learning.language_manipulation import plain_extension,aleph_extension
from loreleai.learning.language_filtering import (
    has_singleton_vars,
    has_duplicated_literal,
)
from loreleai.learning.eval_functions import EvalFunction, Coverage
from loreleai.learning.language_manipulation import variable_instantiation
from loreleai.learning.utilities import (
    compute_bottom_clause, 
    find_allowed_positions, 
    find_allowed_reflexivity, 
    find_frequent_constants)


class Aleph(TemplateLearner):
    """
    Implements the Aleph learner in loreleai. See https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html#SEC45.
    Aleph efficiently searches the hypothesis space by bounding the search from above (X :- true) and below (using the bottom clause),
    and by using mode declarations for predicates. It iteratively adds new clauses that maximize the evalfn. Searching for a new clause
    is done using a branch-and-bound algorithm, where clauses that are guaranteed to not lead to improvements are immediately pruned.

    Aleph currently only supports eval functions that can define an upper bound on the quality of a clause, such as Coverage
    and Compression.
    """

    def __init__(
        self,
        solver: LPSolver,
        eval_fn: EvalFunction,
        max_body_literals=5,
        do_print=False,
    ):
        super().__init__(solver, eval_fn, do_print)
        self._max_body_literals = max_body_literals

    def learn(
        self, examples: Task, knowledge: Knowledge, hypothesis_space: HypothesisSpace, 
        initial_clause: typing.Union[Body,Clause] = None, minimum_freq: int = 0
    ):
        """
        To find a hypothesis, Aleph uses the following set covering approach:
        1.  Select a positive example to be generalised. If none exists, stop; otherwise proceed to the next step.
        2.  Construct the most specific clause (the bottom clause) (Muggleton, 1995) that entails the selected example
            and that is consistent with the mode declarations.
        3.  Search for a clause more general than the bottom clause and that has the best score.
        4.  Add the clause to the current hypothesis and remove all the examples made redundant by it.
        Return to step 1.
        (Description from Cropper and Dumancic )
        """

        # Variables for learning statics
        start_time = datetime.datetime.now()
        i = 0
        stop = False
        self._learnresult = LearnResult()   # Reset in case the learner is reused
        self._prolog_queries = 0
        self._intermediate_coverage = []
        self._eval_fn._clauses_evaluated = 0

        # Assert all BK into engines
        self._solver.retract_all()
        self._assert_knowledge(knowledge)

        # Start with all examples
        examples_to_use = examples
        pos, _ = examples_to_use.get_examples()

        # List of clauses we're learning
        prog = []

        # parameters for aleph_extension()
        allowed_positions = find_allowed_positions(knowledge)
        allowed_reflexivity = find_allowed_reflexivity(knowledge)
        if minimum_freq > 0:
            allowed_constants = find_frequent_constants(knowledge,minimum_freq)
        else:
            allowed_constants = None

        while len(pos) > 0 and not stop:
            i += 1

            # Pick example from pos
            pos_ex = Clause(list(pos)[0], [])
            bk = knowledge.as_clauses()
            bottom = compute_bottom_clause(bk, pos_ex)
            if self._print:
                print("Next iteration: generalizing example {}".format(str(pos_ex)))
                # print("Bottom clause: " + str(bottom))

            # Predicates can only be picked from the body of the bottom clause
            body_predicates = list(
                set(map(
                    lambda l: l.get_predicate(), 
                    bottom.get_body().get_literals()))
            )

            # Constants can only be picked from the literals in the bottom clause,
            # and from constants that are frequent enough in bk (if applicable)
            if allowed_constants is None:
                allowed = lambda l: isinstance(l,Constant) or isinstance(l,int)
            else:
                allowed = lambda l: (isinstance(l,Constant) and l in allowed_constants) or isinstance(l,int)

            constants = list(set(list(filter(
                allowed,
                bottom.get_body().get_arguments(),))))
            if self._print:
                print("Constants in bottom clause: {}".format(constants))
                print("Predicates in bottom clause: {}".format(body_predicates))

            # IMPORTANT: use VALUES of pred and constants, not the variables
            # Has something to do with closures 
            extensions = [
                lambda x,y=pred,z=constants: aleph_extension(x,y,allowed_positions,z,allowed_reflexivity) for pred in body_predicates
            ]

            # Create hypothesis space
            hs = TopDownHypothesisSpace(
                primitives=extensions,
                head_constructor=pos_ex.get_head().get_predicate(),
                expansion_hooks_reject=[
                    #lambda x, y: has_singleton_vars(x, y),
                    lambda x, y: has_duplicated_literal(x, y),
                ],initial_clause=initial_clause
            )

            # Learn 1 clause and add to program
            cl = self._learn_one_clause(examples_to_use, hs)
            prog.append(cl)
            if self._print:
                print("- New clause: " + str(cl))


            # update covered positive examples
            covered = self._execute_program(cl)
            if self._print:
                print(
                    "Clause covers {} pos examples: {}".format(
                        len(pos.intersection(covered)), pos.intersection(covered)
                    )
                )

            # Find intermediate quality of program at this point, add to learnresult (don't cound these as Prolog queries)
            c = set()
            for cl in prog:
                c = c.union(self._execute_program(cl,count_as_query=False))
            pos_covered = len(c.intersection(examples._positive_examples))
            neg_covered = len(c.intersection(examples._negative_examples))
            self._intermediate_coverage.append((pos_covered,neg_covered))

            # Remove covered examples and start next iteration
            pos, neg = examples_to_use.get_examples()
            pos = pos.difference(covered)
            examples_to_use = Task(pos, neg)

            if self._print:
                print("Finished iteration {}".format(i))
                # print("Current program: {}".format(str(prog)))

        # Wrap results into learnresult and return
        self._learnresult['learner'] = "Aleph"
        self._learnresult["total_time"] = (datetime.datetime.now() - start_time).total_seconds()
        self._learnresult["final_program"] = prog
        self._learnresult["num_iterations"] = i
        self._learnresult["evalfn_evaluations"] = self._eval_fn._clauses_evaluated
        self._learnresult["prolog_queries"] = self._prolog_queries
        self._learnresult["intermediate_coverage"] = self._intermediate_coverage

        return self._learnresult

    def initialise_pool(self):
        self._candidate_pool = OrderedSet()

    def put_into_pool(
        self, candidates: Tuple[typing.Union[Clause, Procedure, typing.Sequence], float]
    ) -> None:
        if isinstance(candidates, Tuple):
            self._candidate_pool.add(candidates)
        else:
            self._candidate_pool |= candidates

    def prune_pool(self, minValue):
        """
        Removes all clauss with upper bound on value < minValue form pool
        """
        self._candidate_pool = OrderedSet(
            [t for t in self._candidate_pool if not t[2] < minValue]
        )

    def get_from_pool(self) -> Clause:
        return self._candidate_pool.pop(0)

    def stop_inner_search(
        self, eval: typing.Union[int, float], examples: Task, clause: Clause
    ) -> bool:
        raise NotImplementedError()

    def process_expansions(
        self,
        examples: Task,
        exps: typing.Sequence[Clause],
        hypothesis_space: TopDownHypothesisSpace,
    ) -> typing.Sequence[Clause]:
        # eliminate every clause with more body literals than allowed
        exps = [cl for cl in exps if len(cl) <= self._max_body_literals]

        # check if every clause has solutions
        exps = [
            (cl, self._solver.has_solution(*cl.get_body().get_literals()))
            for cl in exps
        ]
        new_exps = []

        for ind in range(len(exps)):
            if exps[ind][1]:
                # keep it if it has solutions
                new_exps.append(exps[ind][0])
                # print(f"Not removed: {exps[ind][0]}")
            else:
                # remove from hypothesis space if it does not
                hypothesis_space.remove(exps[ind][0])
                # print(f"Removed: {exps[ind][0]}")

        return new_exps

    def _execute_program(self, clause: Clause, count_as_query: bool = True) -> typing.Sequence[Atom]:
        """
        Evaluates a clause using the Prolog engine and background knowledge

        Returns a set of atoms that the clause covers
        """
        if len(clause.get_body().get_literals()) == 0:
            # Covers all possible examples because trivial hypothesis
            return None
        else:
            head_predicate = clause.get_head().get_predicate()
            head_args = clause.get_head_arguments()
            # print("{}({})".format(head_predicate, *head_args))

            sols = self._solver.query(*clause.get_body().get_literals())
            self._prolog_queries += 1 if count_as_query else 0

            # Build a solution by substituting Variables with their found value
            # and copying constants without change
            sols = [head_predicate(*[s[v] if isinstance(v,Variable) else v for v in head_args]) for s in sols]

            return sols

    def _learn_one_clause(
        self, examples: Task, hypothesis_space: TopDownHypothesisSpace
    ) -> Clause:
        """
        Learns a single clause to add to the theory.
        Algorithm from https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html#SEC45
        """
        # reset the search space
        hypothesis_space.reset_pointer()

        # empty the pool just in case
        self.initialise_pool()

        # Add first clauses into pool (active)
        initial_clauses = hypothesis_space.get_current_candidate()
        self.put_into_pool(
            [
                (cl, self.evaluate(examples, cl,hypothesis_space)[0], self.evaluate(examples, cl,hypothesis_space)[1])
                for cl in initial_clauses
            ]
        )
        # print(self._candidate_pool)
        currentbest = None
        currentbestvalue = -99999

        i = 0

        while len(self._candidate_pool) > 0:
            # Optimise: pick smart according to evalFn (e.g. shorter clause when using compression)
            k = self.get_from_pool()
            if self._print:
                print("Expanding clause {}".format(k[0]))
            # Generate children of k
            new_clauses = hypothesis_space.expand(k[0])

            # Remove clauses that are too long...
            new_clauses = self.process_expansions(
                examples, new_clauses, hypothesis_space
            )
            # Compute costs for these children
            value = {cl: self.evaluate(examples, cl, hypothesis_space)[0] for cl in new_clauses}
            upperbound_value = {
                cl: self.evaluate(examples, cl, hypothesis_space)[1] for cl in new_clauses
            }
           
            #print("new_clauses: {}, {}".format(len(new_clauses),[(cl,value[cl]) for cl in new_clauses]))

            for c in new_clauses:
                # If upper bound too low, don't bother expanding
                if upperbound_value[c] <= currentbestvalue and not c == currentbest:
                    hypothesis_space.remove(c)
                else:
                    if value[c] > currentbestvalue:
                        currentbestvalue = value[c]
                        currentbest = c
                        len_before = len(self._candidate_pool)
                        self.prune_pool(value[c])
                        len_after = len(self._candidate_pool)

                        if self._print:
                            print("Found new best: {}: {} {}".format(c,self._eval_fn.name(),value[c]))
                            print("Pruning to upperbound {} >= {}: {} of {} clauses removed".format(self._eval_fn.name(),value[c],(len_before-len_after),len_before))

                    self.put_into_pool((c, value[c], upperbound_value[c]))
                    if self._print:
                        print("Put {} into pool, contains {} clauses".format(str(c),len(self._candidate_pool)))

            i += 1

        if self._print:
            print("New clause: {} with score {}".format(currentbest,currentbestvalue))
        return currentbest
