from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Sequence, Set


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

    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash(self.__repr__())

        return self._hash_cache


@dataclass
class Atom(Formula):

    def __init__(self, predicate: Predicate, arguments: List[Term]):
        super(Atom, self).__init__()
        self.predicate = predicate
        self.arguments = arguments
        self.arg_signature = []

    def substitute(self, term_map: Dict[Term, Term]):
        return global_context.atom(self.predicate, [term_map[x] if x in term_map else x for x in self.arguments])

    def get_predicate(self) -> Predicate:
        return self.predicate

    def get_predicates(self) -> Set[Predicate]:
        return {self.get_predicate()}

    def get_variables(self) -> List[Variable]:
        return [x for x in self.arguments if isinstance(x, Variable)]

    def get_terms(self) -> List[Term]:
        return [x for x in self.arguments]

    def __repr__(self):
        return "{}({})".format(self.predicate.get_name(), ','.join([str(x) for x in self.arguments]))

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.predicate == other.predicate and self.arguments == other.arguments  # _are_two_set_of_literals_identical([self], [other])
        else:
            return False

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

    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash(self.__repr__())

        return self._hash_cache


class Theory:

    def __init__(self, formulas: Sequence[Formula]):
        self._formulas = formulas

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
        raise Exception('Not implemented yet!')


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


class Context:

    def __init__(self):
        self._predicates = {}       # name/arity -> Predicate
        self._variables = {}        # domain -> {name -> Variable}
        self._constants = {}        # domain -> {name -> Constant}
        self._atoms = {}            # Predicate -> { tuple of terms -> Atom}
        self._domains = {'thing': Type('thing')}   # name -> Type

    def _predicate_sig(self, name, arity):
        return f"{name}/{arity}"

    def predicate(self, name, arity, domains=()) -> Predicate:
        if len(domains) == 0:
            domains = [self._domains['thing']]*arity

        if not self._predicate_sig(name, arity) is self._predicates:
            self._predicates[self._predicate_sig(name, arity)] = Predicate(name, arity, domains)

        return self._predicates[self._predicate_sig(name, arity)]

    def variable(self, name, domain=None) -> Variable:
        if domain is None:
            domain = 'thing'

        if domain not in self._variables:
            self._variables[domain] = {}

        if name not in self._variables[domain]:
            self._variables[domain][name] = Variable(name, sym_type=domain)

        return self._variables[domain][name]

    def constant(self, name, domain=None) -> Constant:
        if domain is None:
            domain = 'thing'

        if domain not in self._constants:
            self._constants[domain] = {}

        if name not in self._constants[domain]:
            self._constants[domain][name] = Constant(name, domain)

        return self._constants[domain][name]

    def atom(self, predicate: Predicate, arguments: List[Term]) -> Atom:
        if predicate not in self._atoms:
            self._atoms[predicate] = {}

        if tuple(arguments) not in self._atoms[predicate]:
            self._atoms[predicate][tuple(arguments)] = Atom(predicate, arguments)

        return self._atoms[predicate][tuple(arguments)]


global_context = Context()

