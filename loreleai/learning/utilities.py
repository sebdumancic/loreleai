from itertools import product, combinations
from typing import Sequence

import networkx as nx

from loreleai.language.lp import Type, c_pred, Body, Atom, Predicate, c_var


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