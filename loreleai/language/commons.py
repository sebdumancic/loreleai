from dataclasses import dataclass
from functools import reduce
from itertools import combinations, product
from typing import Dict, List, Tuple, Sequence, Set, Union, Iterator

import kanren
import networkx as nx
import z3

from .utils import MUZ, LP, FOL, KANREN_LOGPY


class Type:
    def __init__(self, name: str):
        self.name = name
        self.elements = set()
        self._engine_objects = {}

    def add(self, elem):
        self.elements.add(elem)

    def remove(self, elem):
        self.elements.remove(elem)

    def add_engine_object(self, elem):
        if z3.is_sort(elem):
            self._engine_objects[MUZ] = elem
        else:
            raise Exception(f"unknown Type object {type(elem)}")

    def get_engine_obj(self, eng):
        assert eng in [MUZ, KANREN_LOGPY]
        return self._engine_objects[eng]

    def as_muz(self):
        return self._engine_objects[MUZ]

    def as_kanren(self):
        raise Exception("types not supported in kanren")

    def __add__(self, other):
        self.add(other)

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())


class Term:
    """
        Term base class. A common base class for Predicate, Constant, Variable and Functor symbols.
    """

    def __init__(self, name, sym_type):
        self.name = name
        self.type = sym_type
        self.hash_cache = None
        self._engine_objects = {}

    def arity(self) -> int:
        """
        Returns the arity of the term

        Returns:
             int
        """
        raise Exception("Not implemented!")

    def get_type(self) -> "Type":
        """
        Returns the type of the term
        """
        return self.type

    def get_name(self) -> str:
        """
        Returns the name of the term

        Return:
            [str]
        """
        return self.name

    def add_engine_object(self, elem) -> None:
        """
        Adds an engine object representing the

        """
        raise NotImplementedError()

    def as_muz(self):
        """
        Returns the object's representation in Z3 Datalog engine (muZ)
        """
        return self._engine_objects[MUZ]

    def as_kanren(self):
        """
        Returns the object's representation in the miniKanren engine
        """
        return self._engine_objects[KANREN_LOGPY]

    def get_engine_obj(self, eng):
        assert eng in [MUZ, KANREN_LOGPY]
        return self._engine_objects[eng]

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.name == other.name and self.type == other.type
        else:
            return False

    def __repr__(self):
        return self.name

    def __hash__(self):
        if self.hash_cache is None:
            self.hash_cache = hash(self.__repr__())
        return self.hash_cache  # hash(self.__repr__())


@dataclass
class Constant(Term):
    """
    Implements a constant in
    """

    def __init__(self, name, sym_type):
        super().__init__(name, sym_type)
        self._id = len(sym_type)
        self.type.add(self)

    def arity(self) -> int:
        return 1

    def id(self) -> int:
        return self._id

    def add_engine_object(self, elem):
        if z3.is_bv_value(elem):
            self._engine_objects[MUZ] = elem
        elif isinstance(elem, str):
            self._engine_objects[KANREN_LOGPY] = elem
        else:
            raise Exception(f"unsupported Constant object {type(elem)}")

    def __repr__(self):
        return self.name

    def __hash__(self):
        if self.hash_cache is None:
            self.hash_cache = hash(self.__repr__())
        return self.hash_cache  # hash(self.__repr__())


@dataclass
class Variable(Term):
    """
    Implements a Variable functionality
    """

    def __init__(self, name: str, sym_type):
        if name[0].islower():
            raise Exception("Variables should uppercase!")
        super().__init__(name, sym_type)

    def arity(self):
        return 1

    def add_engine_object(self, elem):
        if z3.is_expr(elem):
            self._engine_objects[MUZ] = elem
        elif isinstance(elem, kanren.Var):
            self._engine_objects[KANREN_LOGPY] = elem
        else:
            raise Exception(f"unsupported Variable object: {type(elem)}")

    def __repr__(self):
        return self.name

    def __hash__(self):
        if self.hash_cache is None:
            self.hash_cache = hash(self.__repr__() + "/" + str(self.type))
        return self.hash_cache  # hash(self.__repr__() + "/" + str(self.type))

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.name == other.name and self.type == other.type
        else:
            return False


@dataclass
class Structure(Term):
    def __init__(self, name: str, sym_type, arguments):
        super(Structure, self).__init__(name, sym_type)
        self.arguments = arguments

    def __repr__(self):
        return "{}({})".format(self.name, ",".join(self.arguments))

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return (
                self.name == other.type
                and len(self.arguments) == len(other.arguments)
                and all([x == y for (x, y) in zip(self.arguments, other.arguments)])
            )
        else:
            return False

    def arity(self):
        return len(self.arguments)

    def add_engine_object(self, elem):
        raise NotImplementedError()


@dataclass
class Predicate:
    def __init__(self, name: str, arity: int, arguments: List[Type] = None):
        self.name = name
        self.arity = arity
        self.argument_types = (
            arguments if arguments else [Type("thing") for _ in range(arity)]
        )
        self.hash_cache = None
        self._engine_objects = {}

    def get_name(self) -> str:
        return self.name

    def get_arity(self) -> int:
        return self.arity

    def get_arg_types(self) -> List[Type]:
        return self.argument_types

    def signature(self) -> Tuple[str, int]:
        return self.name, self.get_arity()

    def add_engine_object(self, elem):
        if isinstance(elem, tuple):
            # add object as (engine name, object)
            assert elem[0] in [MUZ, KANREN_LOGPY]
            self._engine_objects[elem[0]] = elem[1]
        elif z3.is_func_decl(elem):
            self._engine_objects[MUZ] = elem
        elif isinstance(elem, kanren.Relation):
            self._engine_objects[KANREN_LOGPY] = elem
        else:
            raise Exception(f"unsupported Predicate object {type(elem)}")

    def get_engine_obj(self, eng):
        assert eng in [MUZ, KANREN_LOGPY]
        return self._engine_objects[eng]

    def as_muz(self):
        return self._engine_objects[MUZ]

    def as_kanren(self):
        return self._engine_objects[KANREN_LOGPY]

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return (
                self.get_name() == other.get_name()
                and self.get_arity() == other.get_arity()
                and all(
                    [
                        x == y
                        for (x, y) in zip(self.argument_types, other.get_arg_types())
                    ]
                )
            )
        else:
            return False

    def __repr__(self):
        return "{}({})".format(
            self.name, ",".join([str(x) for x in self.argument_types])
        )

    def __hash__(self):
        if self.hash_cache is None:
            self.hash_cache = hash(self.__repr__())
        return self.hash_cache

    def _map_to_object(self, name: str, arg_position: int) -> Union[Constant, Variable, Structure]:
        if '(' in name:
            raise Exception("automatically converting to structure not yet supported")
        elif name.islower():
            return c_const(name, self.argument_types[arg_position])
        elif name.isupper():
            return c_var(name, self.argument_types[arg_position])
        else:
            raise Exception(f"don't know how to parse {name} to object")

    def __call__(self, *args, **kwargs):
        assert len(args) == self.get_arity()
        assert all([isinstance(x, (Constant, Variable, Structure, str)) for x in args])
        global global_context

        args = [x if isinstance(x, (Constant, Variable, Structure)) else self._map_to_object(x, ind) for ind, x in enumerate(args)]

        if global_context.get_logic() == LP:
            return Literal(self, list(args))
        else:
            raise Exception("FOL not supported yet!")


class Formula:
    def __init__(self):
        self._properties = {}
        self._hash_cache = None

    def substitute(self, term_map: Dict[Term, Term]):
        raise Exception("Not implemented yet!")

    def get_variables(self):
        raise Exception("Not implemented yet!")

    def get_terms(self):
        raise Exception("Not implemented yet!")

    def get_predicates(self) -> Set[Predicate]:
        raise Exception("Not implemented yet!")

    def add_property(self, property_name: str, value):
        self._properties[property_name] = value

    def get_property(self, property_name: str):
        return self._properties.get(property_name, None)

    def has_singleton_var(self) -> bool:
        raise Exception("Not implemented yet!")

    def as_muz(self):
        raise NotImplementedError()

    def as_kanren(self, base_case_recursion=None):
        raise NotImplementedError()

    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash(self.__repr__())

        return self._hash_cache


@dataclass
class Not(Formula):
    def __init__(self, formula: Formula):
        super(Not, self).__init__()
        self.formula = formula

    def substitute(self, term_map: Dict[Term, Term]):
        return Not(self.formula.substitute(term_map))

    def get_variables(self) -> List[Variable]:
        return self.formula.get_variables()

    def get_terms(self) -> List[Term]:
        return self.formula.get_terms()

    def get_formula(self) -> Formula:
        return self.formula

    def get_predicates(self) -> Set[Predicate]:
        return self.formula.get_predicates()

    def as_muz(self):
        return z3.Not(self.formula.as_muz())

    def as_kanren(self, base_case_recursion=None):
        raise Exception("miniKanren does not support negation")

    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash(self.__repr__())

        return self._hash_cache


@dataclass
class Literal(Formula):
    def __init__(self, predicate: Predicate, arguments: List[Term]):
        super(Literal, self).__init__()
        self.predicate = predicate
        self.arguments = arguments
        self.arg_signature = []

    def substitute(self, term_map: Dict[Term, Term]):
        return c_literal(
            self.predicate,
            [term_map[x] if x in term_map else x for x in self.arguments],
        )

    def get_predicate(self) -> Predicate:
        return self.predicate

    def get_predicates(self) -> Set[Predicate]:
        return {self.get_predicate()}

    def get_variables(self) -> List[Variable]:
        return [x for x in self.arguments if isinstance(x, Variable)]

    def get_terms(self) -> List[Term]:
        return [x for x in self.arguments]

    def as_muz(self):
        args = [x.as_muz() for x in self.arguments]
        return self.predicate.as_muz()(*args)

    def as_kanren(self, base_case_recursion=None):
        # not used here, provides base cases forthe recursion
        args = [x.as_kanren() for x in self.arguments]
        return self.predicate.as_kanren()(*args)

    def __repr__(self):
        return "{}({})".format(
            self.predicate.get_name(), ",".join([str(x) for x in self.arguments])
        )

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return (
                self.predicate == other.predicate and self.arguments == other.arguments
            )
        else:
            return False

    def __and__(self, other) -> "Body":
        return Body(self, other)

    def __le__(self, other: Union["Literal", "Body"]) -> "Clause":
        if isinstance(other, Body):
            return Clause(self, other)
        else:
            return Clause(self, [other])

    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash(self.__repr__())

        return self._hash_cache


@dataclass
class Body:
    def __init__(self, *literals):
        self._literals = list(literals)

    def get_literals(self):
        return self._literals

    def __and__(self, other) -> "Body":
        if isinstance(other, Literal):
            self._literals += [other]
            return self
        elif isinstance(other, Body):
            self._literals += other.get_literals()
            return self
        else:
            raise Exception(
                f"Body can be constructed only with Atom or Body, not {type(other)}"
            )


@dataclass
class Clause(Formula):
    """
    Implements the clause functionality

    Args:
        head (Literal): head atom of the clause
        body (List(Atom)): list of atoms in the body of the clause
    """

    def __init__(self, head: Literal, body: Union[List[Literal], Body]):
        super(Clause, self).__init__()
        self._head: Literal = head

        if isinstance(body, Body):
            self._body: Sequence[Literal] = body.get_literals()
        else:
            self._body: Sequence[Literal] = body
        self._body = self._get_atom_order()
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
        return Clause(
            self._head.substitute(term_map),
            list(map(lambda x: x.substitute(term_map), self._body)),
        )

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

    def get_atoms(self, with_predicates: Set[Predicate] = None) -> List[Literal]:
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

    def get_term_signatures(self):
        if self.term_signatures is None:
            self.term_signatures = _create_term_signatures(self._body)
            self.inverted_term_signatures = dict(
                [(frozenset(v.items()), k) for k, v in self.term_signatures.items()]
            )

        return self.term_signatures

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

    def _check_for_unification_with_body(
        self, literals: List[Union[Literal, Not]]
    ) -> List[Dict[Term, Term]]:
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
            self.inverted_term_signatures = dict(
                [(frozenset(v.items()), k) for k, v in self.term_signatures.items()]
            )

        test_clause_literals = _create_term_signatures(literals)
        clause_literals = self.inverted_term_signatures

        test_clause_literals = dict(
            [(frozenset(v.items()), k) for k, v in test_clause_literals.items()]
        )
        matches = dict(
            [
                (clause_literals[x], test_clause_literals[x])
                for x in (clause_literals.keys() & test_clause_literals.keys())
            ]
        )

        if len(matches) < len(clause_literals):
            return [{}]
        elif len(matches) == len(clause_literals):
            return [matches]
        else:
            raise Exception("Multiple unifications possible: not implemented yet!")

    def is_part_of(
        self, clause: "Clause"
    ) -> List[Tuple[List[Literal], Dict[Term, Term]]]:
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
            elif (
                len(self) == len(clause)
                and self.get_predicates() != clause.get_predicates()
            ):
                return []
            else:
                found_substitutions = []
                # construct potential sub-formulas that can be matched
                matching_literals = clause.get_atoms(
                    with_predicates=self.get_predicates()
                )
                for comb in combinations(matching_literals, len(self)):
                    if not are_variables_connected(comb):
                        continue
                    comb = list(comb)
                    answer = self._check_for_unification_with_body(comb)
                    found_substitutions += [(comb, x) for x in answer if len(x)]

                return found_substitutions
        else:
            return []

    def substitute_atoms(
        self,
        atoms_to_replace: List[Union[Literal, Not]],
        new_atom: Literal,
        substitutes: Dict[Term, Term],
    ) -> "Clause":
        """
        Substitutes some atoms in the body with a new atoms

        Args:
            atoms_to_replace (list[Literal]): atom to replace in the clause
            new_atom (Literal): atom to use as the replacement
            substitutes (Dict[Term, Term]): terms substitutes to use in the new atom
        """
        return Clause(
            self._head,
            [new_atom.substitute(substitutes)]
            + [x for x in self._body if x not in atoms_to_replace],
        )

    def unfold_with(
        self, clauses: Union["Clause", Iterator["Clause"]]
    ) -> Iterator["Clause"]:
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
            matching_clauses = [
                x
                for x in clauses
                if x.get_head().get_predicate() == atm.get_predicate()
            ]

            if atm.get_predicate() == self._head.get_predicate():
                # if recursive literals, just leave it in the body
                matching_clauses = []

            # rename variables in all matching clauses
            renamed_clauses = []
            for cl_ind, cl in enumerate(matching_clauses):
                var_map = {}

                for v in cl.get_variables():
                    alternative_name = f"{v.get_name()}{atm_ind}_{cl_ind}"
                    cnt = 1

                    # if the same name appears in the rest of the clause; happens with recursive unfolding
                    if alternative_name in _forbidden_var_names:
                        alternative_name = alternative_name + f"-{cnt}"
                        while alternative_name in _forbidden_var_names:
                            alternative_name = alternative_name.split("-")[0]
                            cnt += 1
                            alternative_name = alternative_name + f"-{cnt}"

                    var_map[v] = c_var(alternative_name, v.get_type())

                renamed_clauses.append(cl.substitute(var_map))

            matching_clauses = renamed_clauses

            if len(matching_clauses):
                candidate_atoms = []

                for mcl in matching_clauses:
                    var_map_matching_clause = dict(
                        zip(mcl.get_head().get_variables(), atm.get_variables())
                    )
                    candidate_atoms.append(
                        [x.substitute(var_map_matching_clause) for x in mcl.get_atoms()]
                    )

                _new_body_atoms.append(candidate_atoms)
            else:
                _new_body_atoms.append([[atm]])

        return [
            Clause(self._head, reduce(lambda u, v: u + v, x))
            for x in product(*_new_body_atoms)
        ]

    def is_recursive(self) -> bool:
        """
        Returns true if the clause is recursive
        """
        return self._head.get_predicate() in [x.get_predicate() for x in self._body]

    def as_muz(self):
        return self._head.as_muz(), [x.as_muz() for x in self._body]

    def as_kanren(self, base_case_recursion=None):
        if self.is_recursive():
            raise Exception(f"recursive rules should not be constructed with .as_kanren() method but should use 'construct_recursive' from kanren package")
        # Should associate a conj goal with the predicate in the head
        # has to be a function
        # rename all variables to make sure there are no strange effects

        # head vars need to be bound to input args of the function
        head_vars = dict([(x, ind) for ind, x in enumerate(self._head.get_variables())])

        # all other arguments need to be bound to their kanren constructs
        other_args = [x.get_terms() for x in self._body]
        other_args = set(reduce(lambda x, y: x + y, other_args, []))
        # remove head variables; these should be bounded to the function arguments
        other_args = [x for x in other_args if x not in head_vars]

        def generic_predicate(*args, core_obj=self, hvars=head_vars, ovars=other_args):
            vars_to_use = dict([(v, kanren.var()) for v in ovars])
            return kanren.conde(
                [x.get_predicate().as_kanren()(
                    *[args[hvars[y]] if y in hvars else vars_to_use[y] for y in x.get_terms()]
                )
                 for x in core_obj.get_atoms()]
            )

        return generic_predicate



    def __contains__(self, item):
        if isinstance(item, Predicate):
            return item.get_name() in map(lambda x: x.predicate.name, self._body)
        elif isinstance(item, Literal):
            return (
                len(
                    [
                        x
                        for x in self._body
                        if x.predicate.get_name() == item.get_predicate().get_name()
                    ]
                )
                > 0
            )
        else:
            return False

    def __add__(self, other: Literal):
        Clause(self._head, self._body + [other])

    def __len__(self):
        return len(self._body)

    def __and__(self, other: Literal):
        self._body += [other]
        self._body = self._get_atom_order()
        return self

    def _get_atom_order(self):
        head_vars = self._head.get_variables()
        all_atoms = [x for x in self._body]
        focus_vars = [head_vars[0]]
        processed_vars = set()
        atom_order = []

        while len(all_atoms) > 0:
            matching_atms = [
                x
                for x in all_atoms
                if any([y in focus_vars for y in x.get_variables()])
            ]
            matching_atms = sorted(
                matching_atms,
                key=lambda x: min(
                    [
                        x.get_variables().index(y) if y in x.get_variables() else 5
                        for y in focus_vars
                    ]
                ),
            )
            processed_vars = processed_vars.union(focus_vars)
            atom_order += matching_atms
            all_atoms = [x for x in all_atoms if x not in matching_atms]
            focus_vars = reduce(
                (lambda x, y: x + y),
                [x.get_variables() for x in matching_atms if x not in processed_vars],
            )

        return atom_order

    def __repr__(self):
        if self._repr_cache is None:
            # head_vars = self._head.get_variables()
            # all_atoms = [x for x in self._body]
            # focus_vars = [head_vars[0]]
            # processed_vars = set()
            # atom_order = []
            #
            # while len(all_atoms) > 0:
            #     matching_atms = [x for x in all_atoms if any([y in focus_vars for y in x.get_variables()])]
            #     matching_atms = sorted(matching_atms, key=lambda x: min([x.get_variables().index(y) if y in x.get_variables() else 5 for y in focus_vars]))
            #     processed_vars = processed_vars.union(focus_vars)
            #     atom_order += matching_atms
            #     all_atoms = [x for x in all_atoms if x not in matching_atms]
            #     focus_vars = reduce((lambda x, y: x + y), [x.get_variables() for x in matching_atms if x not in processed_vars])

            self._repr_cache = "{} :- {}".format(
                self._head, ",".join([str(x) for x in self._body])
            )
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
            bodies = [
                f"{x.get_predicate().get_name()}({','.join([str(var_map[t]) if t in var_map else str(t) for t in x.get_terms()])})"
                for x in self._body
            ]
            bodies = ",".join(bodies)

            self._hash_cache = hash(f"{head_rep} :- {bodies}")

        return self._hash_cache  # hash(self.__repr__())


class Theory:
    def __init__(self, formulas: Sequence[Formula]):
        self._formulas: Sequence = formulas

    def get_formulas(self, predicates: Set[Predicate] = None) -> Sequence[Formula]:
        if predicates:
            return [x for x in self._formulas if any([p for p in x.get_predicates()])]
        else:
            return self._formulas

    def __len__(self):
        return len(self.get_formulas())

    def num_literals(self):
        return sum([len(x) for x in self._formulas])

    def get_predicates(self) -> Set[Predicate]:
        raise Exception("Not implemented yet!")


class Context:
    def __init__(self):
        self._logic = LP
        self._predicates = {}  # name/arity -> Predicate
        self._variables = {}  # domain -> {name -> Variable}
        self._constants = {}  # domain -> {name -> Constant}
        self._literals = {}  # Predicate -> { tuple of terms -> Atom}
                             # ; TO BE USED FOR LP
        self._fatoms = {}  # TO BE USED WITH FOL
        self._domains = {"thing": Type("thing")}  # name -> Type
        self._id_to_constant = {}  # domain (str) -> {id -> Constant}

    def _predicate_sig(self, name, arity):
        return f"{name}/{arity}"

    def get_logic(self):
        return self._logic

    def set_logic(self, logic):
        assert logic in [LP, FOL]
        self._logic = logic

    def get_predicates(self) -> Sequence[Predicate]:
        return [v for k, v in self._predicates.items()]

    def get_constants(self) -> Sequence[Constant]:
        p = [[v for k,v in self._constants[z].items()] for z in self._constants]
        return reduce(lambda x,y: x + y, p, [])

    def get_variables(self) -> Sequence[Variable]:
        return reduce(lambda x, y: x + y, [[v for k, v in self._variables[z].items()] for z in self._variables], [])

    def get_types(self) -> Sequence[Type]:
        return [v for k, v in self._domains.items()]

    def constant_by_id(self, c_id: int, c_type: Union[str, Type]) -> Constant:
        if isinstance(c_type, Type):
            c_type = c_type.name

        return self._id_to_constant[c_type][c_id]

    def type(self, name):
        if name not in self._domains:
            t = Type(name)
            self._domains[name] = t

        return self._domains[name]

    def predicate(self, name, arity, domains=()) -> Predicate:
        if len(domains) == 0:
            domains = [self._domains["thing"]] * arity

        domains = [d if isinstance(d, Type) else self._domains[d] for d in domains]

        if not self._predicate_sig(name, arity) is self._predicates:
            p = Predicate(name, arity, domains)
            self._predicates[self._predicate_sig(name, arity)] = p

        return self._predicates[self._predicate_sig(name, arity)]

    def variable(self, name, domain=None) -> Variable:
        if domain is None:
            domain = "thing"
        elif isinstance(domain, Type):
            domain = domain.name

        if domain not in self._variables:
            self._variables[domain] = {}

        if name not in self._variables[domain]:
            v = Variable(name, sym_type=self._domains[domain])
            self._variables[domain][name] = v

        return self._variables[domain][name]

    def constant(self, name, domain=None) -> Constant:
        if domain is None:
            domain = "thing"
        elif isinstance(domain, Type):
            domain = domain.name

        if domain not in self._id_to_constant:
            self._id_to_constant[domain] = {}

        if domain not in self._constants:
            self._constants[domain] = {}

        if name not in self._constants[domain]:
            c = Constant(name, self._domains[domain])
            self._constants[domain][name] = c
            self._id_to_constant[domain][c.id()] = c

        return self._constants[domain][name]

    def literal(self, predicate: Predicate, arguments: List[Term]) -> "Literal":
        if predicate not in self._literals:
            self._literals[predicate] = {}

        if tuple(arguments) not in self._literals[predicate]:
            self._literals[predicate][tuple(arguments)] = Literal(predicate, arguments)

        return self._literals[predicate][tuple(arguments)]


global_context = Context()


def _get_proper_context(ctx) -> Context:
    if ctx is None:
        global global_context
        return global_context
    else:
        return ctx


def c_pred(name, arity, domains=(), ctx: Context = None) -> Predicate:
    ctx = _get_proper_context(ctx)
    return ctx.predicate(name, arity, domains=domains)


def c_const(name, domain=None, ctx: Context = None) -> Constant:
    ctx = _get_proper_context(ctx)
    return ctx.constant(name, domain=domain)


def c_id_to_const(id: int, type: Union[str, Type], ctx: Context = None) -> Constant:
    ctx = _get_proper_context(ctx)
    return ctx.constant_by_id(id, type)


def c_var(name, domain=None, ctx: Context = None) -> Variable:
    ctx = _get_proper_context(ctx)
    return ctx.variable(name, domain=domain)


def c_literal(
    predicate: Predicate, arguments: List[Term], ctx: Context = None
) -> Literal:
    ctx = _get_proper_context(ctx)
    return ctx.literal(predicate, arguments)


def set_logic(logic, ctx: Context = None):
    ctx = _get_proper_context(ctx)
    ctx.set_logic(logic)


def are_variables_connected(atoms: Sequence[Literal]):
    """
    Checks whether the Variables in the clause are connected

    Args:
        atoms (Sequence[Literal]): atoms whose variables have to be checked

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


def _are_two_set_of_literals_identical(
    clause1: Union[List[Literal], Dict[Sequence[Predicate], Dict]],
    clause2: Union[List[Literal], Dict[Sequence[Predicate], Dict]],
) -> bool:
    """
    Checks whether two sets of literal are identical, i.e. unify, up to the variable naming
    :param clause1:
    :param clause2:
    :return:
    """
    clause1_sig = (
        _create_term_signatures(clause1) if isinstance(clause1, list) else clause1
    )
    clause2_sig = (
        _create_term_signatures(clause2) if isinstance(clause2, list) else clause2
    )

    if len(clause1_sig) != len(clause2_sig):
        return False
    else:
        clause1_sig = dict([(frozenset(v.items()), k) for k, v in clause1_sig.items()])
        clause2_sig = dict([(frozenset(v.items()), k) for k, v in clause2_sig.items()])

        matches = clause1_sig.keys() & clause2_sig.keys()

        # TODO: this is wrong if constants are used
        # terms_cl1 = set()
        # for l in clause1:
        #     for v in l.get_terms():
        #         terms_cl1.add(v)
        #
        # terms_cl2 = set()
        # for l in clause2:
        #     for v in l.get_terms():
        #         terms_cl2.add(v)

        return len(matches) == max(len(clause1_sig), len(clause2_sig))


def _create_term_signatures(
    literals: List[Union[Literal, Not]]
) -> Dict[Term, Dict[Tuple[Predicate], int]]:
    """
        Creates a term signature for each term in the set of literals

        A term signature is a list of all appearances of the term in the clause.
        The appearances are described as a tuple (predicate name, position of the term in the arguments)

        Args:
            literals (List[Literal]): list of literals of the clause

        Returns:
            returns a dictionary with the tuples as keys and their corresponding number of occurrences in the clause

    """
    term_signatures = {}

    for lit in literals:
        for ind, trm in enumerate(lit.get_terms()):
            if trm not in term_signatures:
                term_signatures[trm] = {}

            if isinstance(lit, Not):
                tmp_atm = lit.get_formula()
                if isinstance(tmp_atm, Literal):
                    tmp_sig = (f"not_{tmp_atm.get_predicate().get_name()}", ind)
                else:
                    raise Exception("Only atom can be negated!")
            else:
                tmp_sig = (lit.get_predicate().get_name(), ind)
            term_signatures[trm][tmp_sig] = term_signatures[trm].get(tmp_sig, 0) + 1

    return term_signatures
