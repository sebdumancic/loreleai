import typing
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple
from orderedset import OrderedSet

from loreleai.learning.abstract_learners import Learner, TemplateLearner
from loreleai.learning import Task, Knowledge, HypothesisSpace, TopDownHypothesisSpace
from loreleai.language.commons import (
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
        self, examples: Task, knowledge: Knowledge, hypothesis_space: HypothesisSpace, initial_clause: typing.Union[Body,Clause] = None,
        minimum_freq: int = 0
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
        self._assert_knowledge(knowledge)

        examples_to_use = examples
        pos, _ = examples_to_use.get_examples()

        # List of clauses we're learning
        prog = []

        i = 1
        stop = False

        # parameters for aleph_extension()
        allowed_positions = find_allowed_positions(knowledge)
        allowed_reflexivity = find_allowed_reflexivity(knowledge)
        if minimum_freq > 0:
            allowed_constants = find_frequent_constants(knowledge,minimum_freq)
        else:
            allowed_constants = None

        while len(pos) > 0 and not stop:
            # Pick example from pos
            pos_ex = Clause(list(pos)[0], [])
            bk = knowledge.as_clauses()
            bottom = self._compute_bottom_clause(bk, pos_ex)
            if self._print:
                print("Next iteration: generalizing example {}".format(str(pos_ex)))
                print("Bottom clause: " + str(bottom))

            # Predicates can only be picked from the body of the bottom clause
            body_predicates = list(
                set(map(
                    lambda l: l.get_predicate(), 
                    bottom.get_body().get_literals()))
            )

            # Constants can only be picked from the literals in the bottom clause,
            # and from constants that are frequent enough in bk (if applicable)
            if allowed_constants is None:
                allowed = lambda l: isinstance(l,Constant)
            else:
                allowed = lambda l: isinstance(l,Constant) and l in allowed_constants

            constants = list(set(list(filter(
                allowed,
                bottom.get_body().get_arguments(),))))
            if self._print:
                print("Constants in bottom clause: {}".format(constants))

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
            pos, neg = examples_to_use.get_examples()
            pos = pos.difference(covered)
            examples_to_use = Task(pos, neg)

            if self._print:
                print("Finished iteration {}".format(i))
                # print("Current program: {}".format(str(prog)))
            i += 1

        return prog

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

    def _execute_program(self, clause: Clause) -> typing.Sequence[Atom]:
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
                (cl, self.evaluate(examples, cl)[0], self.evaluate(examples, cl)[1])
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
            value = {cl: self.evaluate(examples, cl)[0] for cl in new_clauses}
            upperbound_value = {
                cl: self.evaluate(examples, cl)[1] for cl in new_clauses
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

    def _compute_bottom_clause(self, theory: Sequence[Clause], c: Clause) -> Clause:
        """
        Computes the bottom clause given a theory and a clause.
        Algorithm from (De Raedt,2008)
        """

        # 1. Find a skolemization substitution θ for c (w.r.t. B and c)
        _, theta = self._skolemize(c)

        # 2. Compute the least Herbrand model M of theory ¬body(c)θ
        body_facts = [
            Clause(l.substitute(theta), []) for l in c.get_body().get_literals()
        ]
        m = herbrand_model(theory + body_facts)

        # 3. Deskolemize the clause head(cθ) <= M and return the result.
        theta_inv = {value: key for key, value in theta.items()}
        return Clause(c.get_head(), [l.get_head().substitute(theta_inv) for l in m])

    def _skolemize(self, clause: Clause) -> Clause:
        # Find all variables in clause
        vars = clause.get_variables()

        # Map from X,Y,Z,... -> sk0,sk1,sk2,...
        subst = {vars[i]: Constant(f"sk{i}", c_type("thing")) for i in range(len(vars))}

        # Apply this substitution to create new clause without quantifiers
        b = []
        h = clause.get_head().substitute(subst)
        for lit in clause.get_body().get_literals():
            b.append(lit.substitute(subst))

        # Return both new clause and mapping
        return Clause(h, b), subst



def _flatten(l) -> Sequence:
    """
    [[1],[2],[3]] -> [1,2,3]
    """
    return [item for sublist in l for item in sublist]


def _all_maps(l1, l2) -> Sequence[Dict[Variable, Constant]]:
    """
    Return all maps between l1 and l2
    such that all elements of l1 have an entry in the map
    """
    sols = []
    for c in combinations_with_replacement(l2, len(l1)):
        sols.append({l1[i]: c[i] for i in range(len(l1))})
    return sols

def herbrand_model(clauses: Sequence[Clause]) -> Sequence[Clause]:
    """
    Computes a minimal Herbrand model of a theory 'clauses'.
    Algorithm from (De Raedt, 2008)
    """
    i = 1
    m = {0: []}
    # Find a fact in the theory (i.e. no body literals)
    facts = list(filter(lambda c: len(c.get_body().get_literals()) == 0, clauses))
    if len(facts) == 0:
        raise AssertionError(
            "Theory does not contain ground facts, which necessary to compute a minimal Herbrand model!"
        )
    # print("Finished iteration 0")
    m[1] = list(facts)
    # print(m[1])
    while Counter(m[i]) != Counter(m[i - 1]):
        model_constants = _flatten(
            [fact.get_head().get_arguments() for fact in m[i]]
        )

        m[i + 1] = []
        rules = list(
            filter(lambda c: len(c.get_body().get_literals()) > 0, clauses)
        )

        for rule in rules:
            # if there is a substition theta such that
            # all literals in rule._body are true in the previous model
            body = rule.get_body()
            body_vars = body.get_variables()
            # Build all substitutions body_vars -> model_constants
            substitutions = _all_maps(body_vars, model_constants)

            for theta in substitutions:
                # add_rule is True unless there is some literal that never
                # occurs in m[i]
                add_fact = True
                for body_lit in body.get_literals():
                    candidate = body_lit.substitute(theta)
                    facts = list(map(lambda x: x.get_head(), m[i]))
                    # print("Does {} occur in {}?".format(candidate,facts))
                    if candidate in facts:
                        pass
                        # print("Yes")
                    else:
                        add_fact = False

                new_fact = Clause(rule.get_head().substitute(theta), [])

                if add_fact and not new_fact in m[i + 1] and not new_fact in m[i]:
                    m[i + 1].append(new_fact)
                    # print("Added fact {} to m[{}]".format(str(new_fact),i+1))
                    # print(m[i+1])

        # print(f"Finished iteration {i}")
        m[i + 1] = list(set(m[i + 1] + m[i]))
        # print("New model: "+str(m[i+1]))
        i += 1
    return m[i]

def find_allowed_positions(knowledge: Knowledge):
    """
    Returns a dict x such that x[constant][predicate] contains
    all positions such i such that `predicate` can have `constant` as
    argument at position i in the background knowledge. 
    This is used to restrict the number of clauses generated through variable 
    instantiation.
    If an atom is not implied by the background theory (i.e. is not in 
    the Herbrand Model), there is no point in considering it, because
    it will never be true.
    """
    facts = herbrand_model(list(knowledge.as_clauses()))
    predicates = set()
    
    # Build dict that will restrict where constants are allowed to appear
    # e.g. allowed_positions[homer][father] = [0,1]
    allowed_positions = dict()
    for atom in facts:
        args = atom.get_head().get_arguments()
        pred = atom.get_head().get_predicate()
        predicates = predicates.union({pred})
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg,Constant):
                # New constant, initialize allowed_positions[constant]
                if not arg in allowed_positions.keys():
                    allowed_positions[arg] = dict()
                # Known constant, but not for this predicate
                if not pred in allowed_positions[arg]:
                    allowed_positions[arg][pred] = [i]
                    # Known constant, and predicate already seen for this constant
                else:
                    if i not in allowed_positions[arg][pred]:
                        allowed_positions[arg][pred] = allowed_positions[arg][pred]+[i]
    
    # Complete dict with empty lists for constant/predicate combinations
    # that were not observed in the background data
    for const in allowed_positions.keys():
        for pred in list(predicates):
            if pred not in allowed_positions[const].keys():
                allowed_positions[const][pred] = []

    return allowed_positions

def find_allowed_reflexivity(knowledge: Knowledge):
    """
    Returns a set of predicates that allow all of its
    arguments to be equal. This is used to prune clauses
    after variable instantiation
    """
    facts = herbrand_model(list(knowledge.as_clauses()))
    allow_reflexivity = set()
    for atom in facts:
        args = atom.get_head().get_arguments()
        pred = atom.get_head().get_predicate()
        if len(args) > 0:
            # If all arguments are equal
            if all(args[i] == args[0] for i in range(len(args))):
                allow_reflexivity.add(pred)
    
    return allow_reflexivity

def find_frequent_constants(knowledge: Knowledge,min_frequency=0):
    facts = herbrand_model(list(knowledge.as_clauses()))
    d = {}

    # Count occurrences of constants
    for atom in facts:
        args = atom.get_head().get_arguments()
        for arg in args: 
            if isinstance(arg, Constant):
                if arg not in d.keys():
                    d[arg] = 0
                else:
                    d[arg] = d[arg] + 1
    
    return [const for const in d.keys() if d[const] >= min_frequency]
    