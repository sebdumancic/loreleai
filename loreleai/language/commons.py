from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Sequence


class Term:
    """
        Term base class. A common base class for Predicate, Constant, Variable and Functor symbols.
    """
    def __init__(self, name, sym_type=None):
        self.name = name
        self.type = sym_type

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.name == other.name and self.type == other.type
        else:
            return False

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.__repr__())

    def arity(self):
        raise Exception('Not implemented!')

    def get_type(self):
        return self.type

    def get_name(self):
        return self.name


@dataclass
class Constant(Term):
    """
    Implements a constant in
    """

    def __init__(self, name, sym_type=None):
        super(Constant, self).__init__(name, sym_type)

    def arity(self):
        return 1


@dataclass
class Variable(Term):
    """
    Implements a Variable functionality
    """

    def __init__(self, name: str, sym_type=None):
        if name[0].islower():
            raise Exception("Variables should uppercase!")
        super(Variable, self).__init__(name, sym_type)

    def arity(self):
        return 1

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.__repr__() + "/" + str(self.type))

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
        return "{}({})".format(self.name, ','.join(self.arguments))

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.name == other.type and \
                   len(self.arguments) == len(other.arguments) and \
                   all([x == y for (x, y) in zip(self.arguments, other.arguments)])
        else:
            return False

    def arity(self):
        return len(self.arguments)


class Type:

    def __init__(self, name: str):
        self.name = name
        self.elements = set()

    def add(self, elem):
        self.elements.add(elem)

    def remove(self, elem):
        self.elements.remove(elem)

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


@dataclass
class Predicate:

    def __init__(self, name: str, arity: int, arguments: List[Type] = None):
        self.name = name
        self.arity = arity
        self.argument_types = arguments if arguments else [Type('thing') for _ in range(arity)]

    def get_name(self) -> str:
        return self.name

    def get_arity(self) -> int:
        return self.arity

    def get_arg_types(self) -> List[Type]:
        return self.argument_types

    def signature(self) -> Tuple[str, int]:
        return self.name, self.get_arity()

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.get_name() == other.get_name() and \
                   self.get_arity() == other.get_arity() and \
                   all([x == y for (x, y) in zip(self.argument_types, other.get_arg_types())])
        else:
            return False

    def __repr__(self):
        return "{}({})".format(self.name, ','.join([str(x) for x in self.argument_types]))

    def __hash__(self):
        return hash(self.__repr__())


class Formula:

    def __init__(self):
        pass

    def substitute(self, term_map: Dict[Term, Term]):
        raise Exception("Not implemented yet!")

    def get_variables(self):
        raise Exception("Not implemented yet!")

    def get_terms(self):
        raise Exception("Not implemented yet!")

    def __hash__(self):
        return hash(self.__repr__())


@dataclass
class Atom(Formula):

    def __init__(self, predicate: Predicate, arguments: List[Term]):
        super(Atom, self).__init__()
        self.predicate = predicate
        self.arguments = arguments
        self.arg_signature = []

    def substitute(self, term_map: Dict[Term, Term]):
        return Atom(self.predicate, [term_map[x] if x in term_map else x for x in self.arguments])

    def get_predicate(self) -> Predicate:
        return self.predicate

    def get_variables(self) -> List[Variable]:
        return [x for x in self.arguments if isinstance(x, Variable)]

    def get_terms(self) -> List[Term]:
        return [x for x in self.arguments]

    def __repr__(self):
        return "{}({})".format(self.predicate.get_name(), ','.join([str(x) for x in self.arguments]))

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return _are_two_set_of_literals_identical([self], [other])
        else:
            return False

    def __hash__(self):
        # if len(self.arg_signature) == 0:
        #     tmp_ind = {}
        #     for ind, v in enumerate(self.arguments):
        #         if v not in tmp_ind:
        #             tmp_ind[v] = ind + 1
        #     self.arg_signature = ",".join([str(tmp_ind[x]) for x in self.arguments])
        return hash(self.__repr__())


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


class Theory:

    def __init__(self, formulas: Sequence[Formula]):
        self._formulas = formulas

    def get_formulas(self) -> Sequence[Formula]:
        return self._formulas

    def __len__(self):
        return len(self.get_formulas())


def _create_term_signatures(literals: List[Union[Atom, Not]]) -> Dict[Term, Dict[Tuple[Predicate], int]]:
    """
        Creates a term signature for each term in the set of literals

        A term signature is a list of all appearances of the term in the clause.
        The appearances are described as a tuple (predicate name, position of the term in the arguments)

        Args:
            literals (List[Atom]): list of literals of the clause

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
                if isinstance(tmp_atm, Atom):
                    tmp_sig = (f'not_{tmp_atm.get_predicate().get_name()}', ind)
                else:
                    raise Exception("Only atom can be negated!")
            else:
                tmp_sig = (lit.get_predicate().get_name(), ind)
            term_signatures[trm][tmp_sig] = term_signatures[trm].get(tmp_sig, 0) + 1

    return term_signatures


def _are_two_set_of_literals_identical(clause1: List[Atom], clause2: List[Atom]) -> bool:
    clause1_sig = _create_term_signatures(clause1)
    clause2_sig = _create_term_signatures(clause2)

    matches = dict([(x, y) for x in clause1_sig for y in clause2_sig
                    if clause1_sig[x] == clause2_sig[y]])

    # TODO: this is wrong if constants are used
    terms = set()
    for l in clause1:
        for v in l.get_terms():
            terms.add(v)
    return len(matches) == len(terms)