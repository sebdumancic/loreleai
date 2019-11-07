from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict, Set, Tuple, Union

from loreleai.language.commons import _create_term_signatures
from ..commons import Atom, Formula, Term, Predicate, Variable, Constant, Not


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
            variables.union(atom.get_variables())

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


def _convert_to_atom(string):
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
        return _convert_to_atom(str)

