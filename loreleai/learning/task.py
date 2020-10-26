from functools import reduce
from typing import Union, List, Sequence

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


class Interpretation:

    def __init__(self, *literals: Atom) -> None:
        self._literals: Sequence[Atom] = literals

    def get_literals(self) -> Sequence[Atom]:
        return self._literals


class Task:

    def __init__(self, *examples: Union[Atom, Interpretation]):
        self._examples: Sequence[Union[Atom, Interpretation]] = examples

    def get_examples(self) -> Sequence[Union[Atom, Interpretation]]:
        return self._examples

    def examples_as_atoms(self) -> Sequence[Atom]:
        if isinstance(self._examples[0], Atom):
            return self._examples
        else:
            raise Exception(f"Cannot return examples as atoms because examples as provided as interpretations")

    def examples_as_interpretations(self) -> Sequence[Interpretation]:
        if isinstance(self._examples[0], Interpretation):
            return self._examples
        else:
            raise Exception(f"Cannot return examples as interpretations because examples as provided as atoms")
