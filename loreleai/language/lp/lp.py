from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import combinations, product
from typing import List, Dict, Set, Tuple, Union, Iterator, Sequence

import networkx as nx
import pygraphviz as pgv

from ..commons import Atom, Formula, Term, Predicate, Variable, Not, Theory, _create_term_signatures, global_context


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
        self._repr_cache = None
        self.term_signatures = None
        self.inverted_term_signatures = None

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
        return Clause(self._head.substitute(term_map), list(map(lambda x: x.substitute(term_map), self._body)))

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

    def has_singleton_var(self) -> bool:
        var_count = {}
        for v in self._head.get_variables():
            if v not in var_count:
                var_count[v] = 0
            var_count[v] += 1

        for atm in self._body:
            for v in atm.get_variables():
                if v not in var_count:
                    var_count[v] = 0
                var_count[v] += 1

        return len([1 for k, v in var_count.items() if v == 1]) > 0

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
        if self.term_signatures is None:
            self.term_signatures = _create_term_signatures(self._body)
            self.inverted_term_signatures = dict([(frozenset(v.items()), k) for k, v in self.term_signatures.items()])

        test_clause_literals = _create_term_signatures(literals)
        clause_literals = self.inverted_term_signatures

        test_clause_literals = dict([(frozenset(v.items()), k) for k, v in test_clause_literals.items()])
        matches = dict([(clause_literals[x], test_clause_literals[x]) for x in (clause_literals.keys() & test_clause_literals.keys())])

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
            a list of tuples:
              - first elements of the tuple is the list of atoms that can be substituted
              -  the second element is the dictionary representing the mapping from variables in self.clause to the
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

        if isinstance(clauses, Clause):
            clauses = [clauses]

        _new_body_atoms = []
        _forbidden_var_names = [x.get_name() for x in self.get_variables()]

        for atm_ind, atm in enumerate(self._body):
            matching_clauses = [x for x in clauses if x.get_head().get_predicate() == atm.get_predicate()]

            if atm.get_predicate() == self._head.get_predicate():
                # if recursive literals, just leave it in the body
                matching_clauses = []

            # rename variables in all matching clauses
            renamed_clauses = []
            for cl_ind, cl in enumerate(matching_clauses):
                var_map = {}

                for v in cl.get_variables():
                    alternative_name = f'{v.get_name()}{atm_ind}_{cl_ind}'
                    cnt = 1

                    # if the same name appears in the rest of the clause; happens with recursive unfolding
                    if alternative_name in _forbidden_var_names:
                        alternative_name = alternative_name + f"-{cnt}"
                        while alternative_name in _forbidden_var_names:
                            alternative_name = alternative_name.split('-')[0]
                            cnt += 1
                            alternative_name = alternative_name + f"-{cnt}"

                    var_map[v] = global_context.variable(alternative_name, v.get_type())

                renamed_clauses.append(cl.substitute(var_map))

            matching_clauses = renamed_clauses

            if len(matching_clauses):
                candidate_atoms = []

                for mcl in matching_clauses:
                    var_map_matching_clause = dict(zip(mcl.get_head().get_variables(), atm.get_variables()))
                    candidate_atoms.append([x.substitute(var_map_matching_clause) for x in mcl.get_atoms()])

                _new_body_atoms.append(candidate_atoms)
            else:
                _new_body_atoms.append([[atm]])

        return [Clause(self._head, reduce(lambda u, v: u + v, x)) for x in product(*_new_body_atoms)]

    def is_recursive(self) -> bool:
        """
        Returns true if the clause is recursive
        """
        return self._head.get_predicate() in [x.get_predicate() for x in self._body]

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
        if self._repr_cache is None:
            head_vars = self._head.get_variables()
            all_atoms = [x for x in self._body]
            focus_vars = [head_vars[0]]
            processed_vars = set()
            atom_order = []

            while len(all_atoms) > 0:
                matching_atms = [x for x in all_atoms if any([y in focus_vars for y in x.get_variables()])]
                matching_atms = sorted(matching_atms, key=lambda x: min([x.get_variables().index(y) if y in x.get_variables() else 5 for y in focus_vars]))
                processed_vars = processed_vars.union(focus_vars)
                atom_order += matching_atms
                all_atoms = [x for x in all_atoms if x not in matching_atms]
                focus_vars = reduce((lambda x, y: x + y), [x.get_variables() for x in matching_atms if x not in processed_vars])

            self._repr_cache = "{} :- {}".format(self._head, ','.join([str(x) for x in atom_order]))
        return self._repr_cache

    def __hash__(self):
        if self._hash_cache is None:
            var_map = {}
            for var in self._head.get_variables():
                if var not in var_map:
                    var_map[var] = len(var_map)

            for atm in self._body:
                for v in atm.get_variables():
                    if v not in var_map:
                        var_map[v] = len(var_map)

            head_rep = f"{self._head.get_predicate().get_name()}({','.join([str(var_map[x] for x in self.get_variables())])})"
            bodies = [f"{x.get_predicate().get_name()}({','.join([str(var_map[t]) if t in var_map else str(t) for t in x.get_terms()])})" for x in self._body]
            bodies = ','.join(bodies)

            self._hash_cache = hash(f"{head_rep} :- {bodies}")

        return self._hash_cache #hash(self.__repr__())


class ClausalTheory(Theory):

    def __init__(self, formulas: Sequence[Clause] = None, read_from_file: str = None):
        assert formulas is not None or read_from_file is not None

        if read_from_file:
            # TODO: fix this for clauses that spread on more than one line
            formulas = []
            inf = open(read_from_file)

            for line in inf.readlines():
                if len(line) > 3 and not line.startswith('#') and not line.startswith('%') and not line.startswith('//') and not line.startswith('true.'):
                    formulas.append(parse(line.strip().replace('.', '')))

        super(ClausalTheory, self).__init__(formulas)

    def get_formulas(self, predicates: Set[Predicate] = None) -> Sequence[Clause]:
        if predicates:
            return [x for x in self._formulas if any([p in predicates for p in x.get_predicates()])]
        else:
            return self._formulas

    def remove_formulas_with_predicates(self, predicates_in_questions: Set[Predicate]):
        """
        Removes all formulas that use at least one of the provided predicates
        """
        self._formulas = [x for x in self._formulas if not any([p in predicates_in_questions for p in x.get_predicates()])]

    def get_predicates(self) -> Set[Predicate]:
        return reduce((lambda x, y: x.union(y)), [x.get_predicates().union({x.get_head().get_predicate()}) for x in self._formulas])

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

        def _unfold_recursively(clause: Clause, clause_index: Dict[Predicate, List[Clause]], forbidden_clauses: Set[Clause]) -> Tuple[List[Clause], Set[Clause]]:
            cl_predicates = [x.get_predicate() for x in clause.get_atoms()]
            if len(forbidden_clauses) == 0:
                matching_clauses_for_unfolding = dict([(k, clause_index[k]) for k in cl_predicates if k in clause_index])
            else:
                matching_clauses_for_unfolding = dict([(k, [p for p in clause_index[k] if p not in forbidden_clauses]) for k in cl_predicates if k in clause_index])

            if len(matching_clauses_for_unfolding) == 0:
                return [clause], set()
            else:
                used_clauses = [v for k, v in matching_clauses_for_unfolding.items()]
                used_clauses = reduce(lambda x, y: x + y, used_clauses)
                _new_form = clause.unfold_with(used_clauses)
                # once the recursive clause is used, do not allow another usage again
                # recursive_clauses = set([x for x in used_clauses if x.is_recursive()])
                # used_clauses = [x for x in used_clauses if not x.is_recursive()]
                # NOT NEEDED ANYMORE BECAUSE RECURSIVE PREDICATES ARE REMOVED FROM THE CANDIDATE SET
                final = [_unfold_recursively(x, clause_index, forbidden_clauses) for x in _new_form]

                final_clauses = reduce(lambda x, y: x + y, [z[0] for z in final])
                final_exclusion = reduce(lambda x, y: x.union(y), [z[1] for z in final])

                return final_clauses, final_exclusion.union(used_clauses)

        # create clause index
        clause_index = {}
        new_set_of_formulas = []
        recursively_defined_predicates = set()

        for cl in self._formulas:
            if cl.is_recursive():
                # detect predicates with recursive definitions
                # do not use them for unfolding because they can remove finite traces
                recursively_defined_predicates.add(cl.get_head().get_predicate())

            head_pred = cl.get_head().get_predicate()
            if head_pred not in clause_index:
                clause_index[head_pred] = []
            clause_index[head_pred].append(cl)

        clauses_to_exclude = set()
        # excluding recursively defined predicates from the candidate set, so that they are not used
        clause_index = dict([(k, v) for k, v in clause_index.items() if k not in recursively_defined_predicates])

        for cl in self.get_formulas():
            if cl in clauses_to_exclude:
                continue

            cls, excls = _unfold_recursively(cl, clause_index, set()) # at the beginning, no forbidden clause (used for recursive ones)
            new_set_of_formulas += cls
            clauses_to_exclude = clauses_to_exclude.union(excls)

        return ClausalTheory(new_set_of_formulas)

    def visualize(self, filename: str, only_numbers=False):
        predicates_in_bodies_only = set()  # names are the predicate names
        predicates_in_heads = set() # names are clauses

        for cl in self._formulas:
            predicates_in_heads.add(cl.get_head().get_predicate())
            predicates_in_bodies_only = predicates_in_bodies_only.union([x.get_predicate() for x in cl.get_atoms()])

        predicates_in_bodies_only = [x for x in predicates_in_bodies_only if x not in predicates_in_heads]

        graph = pgv.AGraph(directed=True)
        cl_to_node_name = {}

        for p in predicates_in_bodies_only:
            cl_to_node_name[p] = len(cl_to_node_name) if only_numbers else f"{p.get_name()}/{p.get_arity()}"
            graph.add_node(cl_to_node_name[p], color='blue')

        for cl in self._formulas:
            ind = len(cl_to_node_name)
            cl_to_node_name[cl] = ind if only_numbers else str(cl)
            cl_to_node_name[cl.get_head().get_predicate()] = ind if only_numbers else str(cl)
            graph.add_node(cl_to_node_name[cl], clause=cl, color='black' if ('latent' in cl.get_head().get_predicate().get_name() or "_" in cl.get_head().get_predicate().get_name()) else 'red')

        for cl in self._formulas:
            body_p = [x.get_predicate() for x in cl.get_atoms()]

            for p in body_p:
                graph.add_edge(cl_to_node_name[cl], cl_to_node_name[p])

        graph.draw(filename, prog='dot')

    def __str__(self):
        return "\n".join([str(x) for x in self._formulas])

    def __len__(self):
        return len(self._formulas)

    def num_literals(self):
        return sum([len(x)+1 for x in self._formulas])


def _convert_to_atom(string: str):
    pred, args = string.strip().replace(')', '').split('(')
    args = args.split(',')

    pred = global_context.predicate(pred, len(args))  # Predicate(pred, len(args))
    # args = [Constant(x) if x.islower() else Variable(x) for x in args]
    args = [global_context.constant(x) if x.islower() else global_context.variable(x) for x in args]

    return global_context.atom(pred, args)


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


