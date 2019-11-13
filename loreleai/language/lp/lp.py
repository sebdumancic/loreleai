from __future__ import annotations

import string
from dataclasses import dataclass
from functools import reduce
from itertools import combinations, product
from typing import List, Dict, Set, Tuple, Union, Iterator, Sequence

import networkx as nx

from loreleai.language.commons import _create_term_signatures
from ..commons import Atom, Formula, Term, Predicate, Variable, Constant, Not, Theory


@dataclass
class Clause(Formula):
    """
    Implements the clause functionality

    Args:
        head (Atom): head atom of the clause
        body (List(Atom)): list of atoms in the body of the clause
    """

    def __init__(self, head: Atom, body: List[Atom]):
        super(Clause, self).__init__()
        self._head = head
        self._body = body
        self._terms = set()

        for lit in self._body:
            self._terms = self._terms.union(lit.get_terms())

    def substitute(self, term_map: Dict[Term, Term]):
        """
            Substitute the terms in the clause

            Args:
                term_map (Dict[Term, Term]): mapping of the terms to their replacements
                                             (key: term from the clause, value: new term to replace it with)

            Return:
                new clause with the replaced literals
        """
        Clause(self._head.substitute(term_map), list(map(lambda x: x.substitute(term_map), self._body)))

    def get_predicates(self) -> Set[Predicate]:
        """
            Returns the predicates in the clause
        """
        return set([x.get_predicate() for x in self._body])

    def get_variables(self) -> Set[Variable]:
        """
            Returns the of variables in the clause
        """
        variables = set()

        for atom in self._body:
            variables = variables.union(atom.get_variables())

        return variables

    def get_atoms(self, with_predicates: Set[Predicate] = None) -> List[Atom]:
        """
            Returns the set of atoms in the clause

            Args:
                with_predicates (Set[Predicates], optional): return only atoms with these predicates
        """
        if with_predicates is None:
            return self._body
        else:
            return [x for x in self._body if x.get_predicate() in with_predicates]

    def get_head(self):
        return self._head

    def _check_for_unification_with_body(self, literals: List[Union[Atom, Not]]) -> List[Dict[Term, Term]]:
        """
        Checks whether the body of the clause unifies with the provided set of literals

        Args:
            literals (List[Union[Atom, Not]]): literals (another clause)

        Returns:
            A list of possible substitutions
                (each substitution is a dictionary where the
                    -- key is the variables from the clause which should be substituted
                    -- the values are the terms from the literals which should be used as substitutions)

        # TODO: check whether everything is correct for working with constants
        """
        test_clause_literals = _create_term_signatures(literals)
        clause_literals = _create_term_signatures(self._body)

        matches = [(x, y) for x in clause_literals for y in test_clause_literals if clause_literals[x] == test_clause_literals[y]]
        matches = dict(matches)

        if len(matches) < len(clause_literals):
            return [{}]
        elif len(matches) == len(clause_literals):
            return [matches]
        else:
            raise Exception("Multiple unifications possible: not implemented yet!")

    def is_part_of(self, clause: Clause) -> List[Tuple[List[Atom], Dict[Term, Term]]]:
        """
        Checks whether the body of (self.)clause unifies with the part of the body of the provided clause

        Args:
            clause (Clause): is self.clause part of this clause?

        Return:
            a list of tuples, where the first elements of the tuple is the list of atoms that can be substituted
            and the second element is the dictionary representing the mapping from variables in self.clause to the
            variables is the provided clause
        """
        if isinstance(self, type(clause)):
            if len(self) > len(clause):
                return []
            elif len(self) == len(clause) and self.get_predicates() != clause.get_predicates():
                return []
            else:
                found_substitutions = []
                # construct potential sub-formulas that can be matched
                matching_literals = clause.get_atoms(with_predicates=self.get_predicates())
                for comb in combinations(matching_literals, len(self)):
                    comb = list(comb)
                    answer = self._check_for_unification_with_body(comb)
                    found_substitutions += [(comb, x) for x in answer]

                return found_substitutions
        else:
            return []

    def substitute_atoms(self, atoms_to_replace: List[Union[Atom, Not]],
                         new_atom: Atom,
                         substitutes: Dict[Term, Term]) -> Clause:
        """
        Substitutes some atoms in the body with a new atoms

        Args:
            atoms_to_replace (list[Atom]): atom to replace in the clause
            new_atom (Atom): atom to use as the replacement
            substitutes (Dict[Term, Term]): terms substitutes to use in the new atom
        """
        return Clause(self._head, [new_atom.substitute(substitutes)] + [x for x in self._body if x not in atoms_to_replace])

    def unfold_with(self, clauses: Union[Clause, Iterator[Clause]]) -> Iterator[Clause]:
        """
        Unfolds the clause with given clauses
            If more than one clause is given for unfolding, assumes no clauses with the same head are provided

        Args:
            clauses [Union[Clause, List[Clauses]]: clauses to use for unfolding

        Returns:
            unfolded clause [Clause]
        """
        def _get_variable_map_unfolding(atm: Atom, clause: Clause, other_var_names_used=()):
            remaining_vars = clause.get_variables()
            head_vars = clause.get_head().get_variables()
            remaining_vars = [x for x in remaining_vars if x not in head_vars]

            existing_var_names = [x.get_name() for x in self.get_variables()]
            remaining_var_subs = zip(remaining_vars, [x for x in string.ascii_uppercase if x not in existing_var_names and x not in other_var_names_used])
            remaining_var_subs = dict([(x, Variable(n, x.get_type())) for x, n in remaining_var_subs])

            vars_subs = dict(zip(head_vars, atm.get_variables()))

            # need them separately because there can be variable overlap, so the remaining vars should be replaced first
            return vars_subs, remaining_var_subs

        def _substitute_vars_in_atoms(atms: Iterator[Atom], head_vars: Dict[Variable, Variable], remaining_vars: Dict[Variable, Variable]):
            atms = [x.substitute(remaining_vars) for x in atms]
            return [x.substitute(head_vars) for x in atms]

        if isinstance(clauses, Clause):
            clauses = [clauses]

        _new_body_atoms = []
        _new_var_names_used_so_far = set()

        for atm in self._body:
            matching_clauses = [x for x in clauses if x.get_head().get_predicate() == atm.get_predicate()]

            if len(matching_clauses):
                candidate_atoms = []

                for mcl in matching_clauses:
                    var_subs_head, var_subs_body = _get_variable_map_unfolding(atm, mcl, _new_var_names_used_so_far)

                    # remember which auxiliary var names were used so far
                    var_names_used_here = [v.get_name() for k, v in var_subs_body.items() if
                                           k not in mcl.get_head().get_variables()]
                    _new_var_names_used_so_far = _new_var_names_used_so_far.union(var_names_used_here)
                    candidate_atoms.append(_substitute_vars_in_atoms(mcl.get_atoms(), var_subs_head, var_subs_body))

                # matching_clauses = matching_clauses[0]
                # var_subs_head, var_subs_body = _get_variable_map_unfolding(atm, matching_clauses, _new_var_names_used_so_far)
                #
                # # remember which auxiliary var names were used so far
                # var_names_used_here = [v.get_name() for k, v in var_subs_body.items() if k not in matching_clauses.get_head().get_variables()]
                # _new_var_names_used_so_far = _new_var_names_used_so_far.union(var_names_used_here)
                #
                # _new_body_atoms += _substitute_vars_in_atoms(matching_clauses.get_atoms(), var_subs_head, var_subs_body)
                _new_body_atoms.append(candidate_atoms)
            else:
                _new_body_atoms.append([[atm]])

        # clauses = []
        #
        # for itm in product(*_new_body_atoms):

        return [Clause(self._head, reduce(lambda u, v: u + v, x)) for x in product(*_new_body_atoms)]

    def __contains__(self, item):
        if isinstance(item, Predicate):
            return item.get_name() in map(lambda x: x.predicate.name, self._body)
        elif isinstance(item, Atom):
            return len([x for x in self._body if x.predicate.get_name() == item.get_predicate().get_name()]) > 0
        else:
            return False

    def __add__(self, other: Atom):
        Clause(self._head, self._body + [other])

    def __len__(self):
        return len(self._body)

    def __repr__(self):
        return "{} :- {}".format(self._head, ','.join([str(x) for x in self._body]))

    def __hash__(self):
        return hash(self.__repr__())


class ClausalTheory(Theory):

    def __init__(self, formulas: Sequence[Clause] = None, read_from_file: str = None):
        assert formulas is not None or read_from_file is not None

        if read_from_file:
            # TODO: fix this for clauses that spread on more than one line
            formulas = []
            inf = open(read_from_file)

            for line in inf.readlines():
                if len(line) > 3 and not line.startswith('#') and not line.startswith('//'):
                    formulas.append(parse(line.strip().replace('.', '')))

        super(ClausalTheory, self).__init__(formulas)

    def get_formulas(self) -> Sequence[Clause]:
        return self._formulas

    def unfold(self):
        """
        Unfolds the theory

        A theory containing two clauses
                h :- d,c,r.
                d :- a,b.
        Would be unfolded into
                h :- a,b,c,r.

        Returns:
             unfolded theory [Theory]
        """

        def _unfold_recursively(clause: Clause, clause_index: Dict[Predicate, List[Clause]]) -> Tuple[List[Clause], Set[Clause]]:
            cl_predicates = [x.get_predicate() for x in clause.get_atoms()]
            matching_clauses_for_unfolding = dict([(k, clause_index[k]) for k in cl_predicates if k in clause_index])

            if len(matching_clauses_for_unfolding) == 0:
                return [clause], set()
            else:
                used_clauses = [v for k, v in matching_clauses_for_unfolding.items()]
                _new_form = clause.unfold_with(reduce(lambda x, y: x + y, used_clauses))
                final = [_unfold_recursively(x, clause_index) for x in _new_form]

                final_clauses = reduce(lambda x, y: x + y, [z[0] for z in final])
                final_exclusion = reduce(lambda x, y: x.union(y), [z[1] for z in final])

                return final_clauses, final_exclusion.union(reduce(lambda x, y: x + y, used_clauses))
            # elif all([len(matching_clauses_for_unfolding[k]) == 1 for k in matching_clauses_for_unfolding]):
            #     used_clauses = [v[0] for k, v in matching_clauses_for_unfolding.items()]
            #     _new_form = clause.unfold_with(used_clauses)
            #
            #     final_clauses = []
            #     final = [_unfold_recursively(x, clause_index) for x in _new_form]
            #
            #     final_clauses = reduce(lambda x, y: x + y, [z[0] for z in final])
            #     final_exclusion = reduce(lambda x, y: x.union(y), [z[1] for z in final])
            #
            #     # final_clauses, final_exclusion = _unfold_recursively(_new_form, clause_index)
            #     return final_clauses, final_exclusion.union(used_clauses)
            # else:
            #     itms_to_use = [v for k, v in matching_clauses_for_unfolding.items()]
            #     prod = product(*itms_to_use, repeat=2 if len(itms_to_use) == 1 else 1)
            #     used_clauses = set(reduce(lambda x, y: x + y, [v for k, v in matching_clauses_for_unfolding.items()]))
            #
            #     _new_set_of_formulas = []
            #     for cmb in prod:
            #         _new_form = clause.unfold_with(cmb)
            #         _new_set_of_formulas.append(_new_form)
            #
            #     #_new_set_of_formulas = [clause.unfold_with(cmb) for cmb in prod]
            #     _new_set_of_formulas = [_unfold_recursively(y, clause_index) for x in _new_set_of_formulas for y in x]
            #
            #     final_clauses = reduce(lambda x, y: x + y, [x[0] for x in _new_set_of_formulas])
            #     final_used = reduce(lambda x, y: x.union(y), [x[1] for x in _new_set_of_formulas])
            #
            #     return final_clauses, final_used.union(used_clauses)

        # create clause index
        clause_index = {}
        new_set_of_formulas = []

        for cl in self._formulas:
            head_pred = cl.get_head().get_predicate()
            if head_pred not in clause_index:
                clause_index[head_pred] = []
            clause_index[head_pred].append(cl)

        clauses_to_exclude = set()

        for cl in self.get_formulas():
            if cl in clauses_to_exclude:
                continue

            cls, excls = _unfold_recursively(cl, clause_index)
            new_set_of_formulas += cls
            clauses_to_exclude = clauses_to_exclude.union(excls)

        return ClausalTheory(new_set_of_formulas)


def _convert_to_atom(string: str):
    pred, args = string.strip().replace(')', '').split('(')
    args = args.split(',')

    pred = Predicate(pred, len(args))
    args = [Constant(x) if x.islower() else Variable(x) for x in args]

    return Atom(pred, args)


def parse(string: str):
    if ":-" in string:
        head, body = string.split(":-")
        head, body = head.strip(), body.strip()
        body = [x + ")" for x in body.split("),")]
        head, body = _convert_to_atom(head), [_convert_to_atom(x) for x in body]
        return Clause(head, body)
    else:
        return _convert_to_atom(string)


def are_variables_connected(atoms: Sequence[Atom]):
    """
    Checks whether the Variables in the clause are connected

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


