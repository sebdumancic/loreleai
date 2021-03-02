from itertools import product, combinations
from typing import Sequence

import networkx as nx

from loreleai.language.lp import Type, c_pred, Clause, Body, Atom, Predicate, c_var, Variable, Constant
from loreleai.learning.task import Knowledge
from typing import Sequence, Dict, Tuple


class FillerPredicate:
    def __init__(
        self,
        prefix_name: str,
        arity: int = None,
        max_arity: int = None,
        min_arity: int = None,
    ):
        assert bool(max_arity) == bool(min_arity) and min_arity <= max_arity
        assert bool(arity) != bool(max_arity)
        self._prefix_name = prefix_name
        self._arity = arity
        self._max_arity = max_arity
        self._min_arity = min_arity
        self._instance_counter = 0

    def new(
        self, arity: int = None, argument_types: Sequence[Type] = None
    ) -> Predicate:
        """
        Creates a new predicate from the template

        if the FillerPredicate is constructed with arity set, it returns a new predicate with that arity
        if the
        """
        assert (
            arity is not None or self._arity is not None or argument_types is not None
        )
        self._instance_counter += 1

        if argument_types is not None:
            assert (
                len(argument_types) == self._arity
                or len(argument_types) >= self._min_arity
                or len(argument_types) <= self._max_arity
            )
            return c_pred(
                f"{self._prefix_name}_{self._instance_counter}",
                self._arity,
                argument_types,
            )
        elif arity is not None:
            return c_pred(f"{self._prefix_name}_{self._instance_counter}", arity)
        else:
            return c_pred(f"{self._prefix_name}_{self._instance_counter}", self._arity)

    def _from_body_fixed_arity(
        self,
        body: Body,
        arity: int = None,
        arg_types: Sequence[Type] = None,
        use_as_head_predicate: Predicate = None,
    ) -> Sequence[Atom]:
        """
        Creates a head atom given the body of the clause
        :param body:
        :param arity: (optional) desired arity if specified with min/max when constructing the FillerPredicate
        :param arg_types: (optional) argument types to use
        :return:
        """
        assert bool(arity) != bool(arg_types)
        vars = body.get_variables()

        if use_as_head_predicate and arg_types is None:
            arg_types = use_as_head_predicate.get_arg_types()

        if arg_types is None:
            base = [vars] * arity
        else:
            matches = {}

            for t_ind in range(len(arg_types)):
                matches[t_ind] = []
                for v_ind in range(len(vars)):
                    if vars[v_ind].get_type() == arg_types[t_ind]:
                        matches[t_ind].append(vars[v_ind])

            base = [matches[x] for x in range(arity)]

        heads = []
        for comb in product(*base):
            self._instance_counter += 1
            if use_as_head_predicate is not None:
                pred = use_as_head_predicate
            elif arg_types is None:
                pred = c_pred(f"{self._prefix_name}_{self._instance_counter}", arity)
            else:
                pred = c_pred(
                    f"{self._prefix_name}_{self._instance_counter}",
                    len(arg_types),
                    arg_types,
                )

            heads.append(Atom(pred, list(comb)))

        return heads

    def new_from_body(
        self,
        body: Body,
        arity: int = None,
        argument_types: Sequence[Type] = None,
        use_as_head_predicate: Predicate = None,
    ) -> Sequence[Atom]:
        """
        Constructs all possible head atoms given a body of a clause

        If use_as_head_predicate is provided, it uses that.
        Then, it check is the FillerPredicate is instantiated with a fixed arity
        Then, it check if the arity argument is provided
        """
        if use_as_head_predicate:
            return self._from_body_fixed_arity(body, arity=use_as_head_predicate.get_arity(), arg_types=use_as_head_predicate.get_arg_types(), use_as_head_predicate=use_as_head_predicate)
        elif self._arity is not None:
            # a specific arity is provided
            return self._from_body_fixed_arity(body, self._arity, argument_types)
        else:
            # min -max arity is provided
            if arity is not None:
                # specific arity is requests
                return self._from_body_fixed_arity(body, arity)
            else:
                heads = []
                for i in range(self._min_arity, self._max_arity + 1):
                    heads += self._from_body_fixed_arity(body, i)

                return heads

    def all_possible_atoms(self) -> Sequence[Atom]:
        """
        Creates all possible argument configurations for the atom
        """
        head_variables = [c_var(chr(x)) for x in range(ord("A"), ord("Z"))][
            : self._arity
        ]
        if self._arity is not None:
            return [
                Atom(self.new(), list(x))
                for x in product(head_variables, repeat=self._arity)
            ]
        else:
            combos = []
            for i in range(self._min_arity, self._max_arity):
                combos += [
                    Atom(self.new(arity=i), list(x))
                    for x in product(head_variables, repeat=i)
                ]
            return combos

    def is_created_by(self, predicate: Predicate) -> bool:
        """
        checks if a given predicate is created by the FillerPredicate

        it does so by checking if the name of the predicate is in [name]_[number] format and the name is
            equal to self._prefix_name and number is <= self._instance_counter
        """
        sp = predicate.get_name().split("_")
        if len(sp) != 2:
            return False
        else:
            if sp[0] == self._prefix_name and int(sp[1]) <= self._instance_counter:
                return True
            else:
                return False

    def _add_to_body_fixed_arity(self, body: Body, arity: int) -> Sequence[Body]:
        new_pred_stash = {}  # arg_types tuple -> pred

        vars = body.get_variables()

        bodies = []

        args = list(combinations(vars, arity))
        for ind in range(len(args)):
            arg_types = (x.get_type() for x in args[ind])

            if arg_types in new_pred_stash:
                pred = new_pred_stash[arg_types]
            else:
                self._instance_counter += 1
                pred = c_pred(f"{self._prefix_name}{self._instance_counter}", arity, arg_types)
                new_pred_stash[arg_types] = pred

            bodies.append(body + pred(*args[ind]))

        return bodies

    def add_to_body(self, body: Body) -> Sequence[Body]:
        """
        Adds the filler predicate to the body

        It is meant to be used to create a recursive clause
        """

        if self._arity:
            return self._add_to_body_fixed_arity(body, self._arity)
        else:
            bodies = []

            for ar in range(self._min_arity, self._max_arity + 1):
                bodies += self._add_to_body_fixed_arity(body, ar)

            return bodies


def are_variables_connected(atoms: Sequence[Atom]):
    """
    Checks whether the Variables in the clause are connected

    Args:
        atoms (Sequence[Atom]): atoms whose variables have to be checked

    """
    g = nx.Graph()

    for atm in atoms:
        vrs = atm.get_variables()
        if len(vrs) == 1:
            g.add_node(vrs[0])
        else:
            for cmb in combinations(vrs, 2):
                g.add_edge(cmb[0], cmb[1])

    res = nx.is_connected(g)
    del g

    return res


def compute_bottom_clause(theory: Sequence[Clause], c: Clause) -> Clause:
    """
    Computes the bottom clause given a theory and a clause.
    Algorithm from (De Raedt,2008)
    """
    # 1. Find a skolemization substitution θ for c (w.r.t. B and c)
    _, theta = skolemize(c)

    # 2. Compute the least Herbrand model M of theory ¬body(c)θ
    body_facts = [
        Clause(l.substitute(theta), []) for l in c.get_body().get_literals()
    ]
    m = herbrand_model(theory + body_facts)

    # 3. Deskolemize the clause head(cθ) <= M and return the result.
    theta_inv = {value: key for key, value in theta.items()}
    return Clause(c.get_head(), [l.get_head().substitute(theta_inv) for l in m])

def skolemize(clause: Clause) -> Clause:
    # Find all variables in clause
    vars = clause.get_variables()

    # Map from X,Y,Z,... -> sk0,sk1,sk2,...
    subst = {vars[i]: Constant(f"sk{i}", c_type("thing")) for i in range(len(vars))}

    # Apply this substitution to create new clause without quantifiers
    return clause.substitute(subst), subst


def herbrand_model(clauses: Sequence[Clause]) -> Sequence[Clause]:
    """
    Computes a minimal Herbrand model of a theory 'clauses'.
    Algorithm from Logical and Relational learning (De Raedt, 2008)
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

    # If all clauses are just facts, there is nothing to be done.
    if len(facts) == len(clauses):
        return clauses

    #BUG: doesn't work properly after pylo update...

    m[1] = list(facts)
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
    Returns the set of predicates in `knowledge` that allow all of its
    arguments to be equal. That is, if `knowledge` contains a fact pred(x,x,x),
    pred will be in the return value.
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
    """
    Returns a list of all constants that occur at least `min_frequency` times in 
    `knowledge`
    """
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