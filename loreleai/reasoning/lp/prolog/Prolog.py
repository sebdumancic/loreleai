from abc import abstractmethod
from typing import Union

from loreleai.language.lp import Clause, Atom, Procedure, Context, Constant, Type, Predicate, Variable
from ..lpsolver import LPSolver


class Prolog(LPSolver):

    @abstractmethod
    def __init__(self, name, knowledge_base=None, background_knowledge=None, ctx: Context = None):
        super().__init__(name, knowledge_base, background_knowledge, ctx)

    def _create_objects(self, ctx: Context):
        """
        No need to do this for Prolog engines
        """
        pass

    def declare_constant(self, elem_constant: Constant) -> None:
        """
        No need to do this for Prolog engines
        """
        pass

    def declare_type(self, elem_type: Type) -> None:
        """
        No need to do this for Prolog engines
        """
        pass

    def declare_predicate(self, elem_predicate: Predicate) -> None:
        """
        No need to do this for Prolog engines
        """
        pass

    def declare_variable(self, elem_variable: Variable) -> None:
        """
        No need to do this for Prolog engines
        """
        pass

    def assert_fact(self, fact: Atom) -> None:
        self.assertz(fact)

    def assert_rule(self, rule: Union[Clause, Procedure]) -> None:
        self.assertz(rule)

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




