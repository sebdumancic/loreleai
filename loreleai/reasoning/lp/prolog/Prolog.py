from abc import ABC, abstractmethod
from typing import Union

from loreleai.language.lp import Clause, Atom, Procedure


class Prolog(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def consult(self, filename: str):
        pass

    @abstractmethod
    def use_module(self, module: str, **kwargs):
        pass

    @abstractmethod
    def asserta(self, clause: Union[Atom, Clause, Procedure]):
        pass

    @abstractmethod
    def assertz(self, clause: Union[Atom, Clause, Procedure]):
        pass

    @abstractmethod
    def retract(selfself, clause: Union[Atom]):
        pass

    @abstractmethod
    def has_solution(self, query):
        pass

    @abstractmethod
    def query(self, *query, **kwargs):
        pass

    @abstractmethod
    def register_foreign(self, pyfunction, arity):
        pass




