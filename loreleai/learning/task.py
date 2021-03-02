from functools import reduce
from typing import Union, List, Sequence, Tuple, Set

from loreleai.language.lp import Atom, Clause, Procedure, Program


class Knowledge:

    def __init__(self, *pieces: Union[Atom, Clause, Procedure, Program]) -> None:
        self._knowledge_pieces: List = reduce(
            lambda x, y: x + [y] if isinstance(y, (Atom, Clause, Procedure)) else x + y.get_clauses(),
            pieces,
            [])

    def _add(self, other: Union[Atom, Clause, Procedure, Program]) -> None:
        if isinstance(other, (Atom, Clause, Procedure)):
            self._knowledge_pieces.append(other)
        else:
            self._knowledge_pieces += other.get_clauses()

    def add(self, *piece: Union[Atom, Clause, Procedure, Program]) -> None:
        map(lambda x: self._add(x), piece)

    def get_all(self):
        return self._knowledge_pieces

    def get_clauses(self):
        return [x for x in self._knowledge_pieces if isinstance(x, (Clause, Procedure))]

    def get_atoms(self):
        return [x for x in self._knowledge_pieces if isinstance(x, Atom)]

    def as_clauses(self):
        l = []
        for x in self.get_all():
            if isinstance(x,Clause):
                l.append(x)
            elif isinstance(x,Atom):
                l.append(Clause(x,[]))
            elif isinstance(x,Procedure):
                for cl in x.get_clauses():
                    l.append(cl)
            elif isinstance(x,Program):
                for cl in x.get_clauses():
                    l.append(cl)
            else:
                raise AssertionError("Knowledge can only contain clauses, atoms, procedures or programs!")
        return l


class Interpretation:

    def __init__(self, *literals: Atom) -> None:
        self._literals: Sequence[Atom] = literals

    def get_literals(self) -> Sequence[Atom]:
        return self._literals


class Task:

    def __init__(self, positive_examples: Set[Atom] = None, negative_examples: Set[Atom] = None, interpretations: Sequence[Interpretation] = None):
        self._examples: Sequence[Interpretation] = interpretations
        self._positive_examples: Set[Atom] = positive_examples
        self._negative_examples: Set[Atom] = negative_examples

    def get_examples(self) -> Union[Sequence[Interpretation], Tuple[Set[Atom], Set[Atom]]]:
        if self._examples is not None:
            return self._examples
        else:
            return self._positive_examples, self._negative_examples
