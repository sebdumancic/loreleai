import typing
from abc import ABC, abstractmethod

from orderedset import OrderedSet

from loreleai.language.lp import c_pred, Clause, Procedure, Atom
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_singleton_vars, has_duplicated_literal
from loreleai.learning.language_manipulation import plain_extension
from loreleai.learning.task import Task, Knowledge
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog

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

    def __init__(self, solver_instance: Prolog):
        self._solver = solver_instance
        self._candidate_pool = []

    def _assert_knowledge(self, knowledge: Knowledge):
        """
        Assert knowledge into Prolog engine
        """
        facts = knowledge.get_atoms()
        for f_ind in range(len(facts)):
            self._solver.assertz(facts[f_ind])

        clauses = knowledge.get_clauses()
        for cl_ind in range(len(clauses)):
            self._solver.assertz(clauses[cl_ind])

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

    @abstractmethod
    def evaluate(self, examples: Task, clause: Clause) -> typing.Union[int, float]:
        """
        Evaluates a clause  of a task

        Returns a number (the higher the better)
        """
        raise NotImplementedError()

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
            # add into pull
            self.put_into_pool(exps)

            score = self.evaluate(examples, current_cand)

        return current_cand

    def learn(self, examples: Task, knowledge: Knowledge, hypothesis_space: TopDownHypothesisSpace):
        """
        General learning loop
        """

        self._assert_knowledge(knowledge)
        final_program = []
        examples_to_use = examples
        pos, _ = examples_to_use.get_examples()

        while len(final_program) == 0 or len(pos) > 0:
            # learn na single clause
            cl = self._learn_one_clause(examples_to_use, hypothesis_space)
            final_program.append(cl)

            # update covered positive examples
            covered = self._execute_program(cl)

            pos, neg = examples_to_use.get_examples()
            pos = pos.difference(covered)

            examples_to_use = Task(pos, neg)

        return final_program


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

    def __init__(self, solver_instance: Prolog, max_body_literals=4):
        super().__init__(solver_instance)
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

    def evaluate(self, examples: Task, clause: Clause) -> typing.Union[int, float]:
        covered = self._execute_program(clause)

        pos, neg = examples.get_examples()

        covered_pos = pos.intersection(covered)
        covered_neg = neg.intersection(covered)

        if len(covered_neg) > 0:
            return 0
        else:
            return len(covered_pos)

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


if __name__ == '__main__':
    # define the predicates
    father = c_pred("father", 2)
    mother = c_pred("mother", 2)
    grandparent = c_pred("grandparent", 2)

    # specify the background knowledge
    background = Knowledge(father("a", "b"), mother("a", "b"), mother("b", "c"),
                           father("e", "f"), father("f", "g"),
                           mother("h", "i"), mother("i", "j"))

    # positive examples
    pos = {grandparent("a", "c"), grandparent("e", "g"), grandparent("h", "j")}

    # negative examples
    neg = {grandparent("a", "b"), grandparent("a", "g"), grandparent("i", "j")}

    task = Task(positive_examples=pos, negative_examples=neg)

    # create Prolog instance
    prolog = SWIProlog()

    learner = SimpleBreadthFirstLearner(prolog, max_body_literals=3)

    # create the hypothesis space
    hs = TopDownHypothesisSpace(primitives=[lambda x: plain_extension(x, father, connected_clauses=True),
                                            lambda x: plain_extension(x, mother, connected_clauses=True)],
                                head_constructor=grandparent,
                                expansion_hooks_reject=[lambda x, y: has_singleton_vars(x, y),
                                                        lambda x, y: has_duplicated_literal(x, y)])

    program = learner.learn(task, background, hs)

    print(program)



