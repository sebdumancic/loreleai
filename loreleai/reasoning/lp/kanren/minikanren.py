from typing import Union, Dict, Sequence, Tuple

import kanren

from loreleai.language import KANREN_LOGPY
from loreleai.language.kanren import Constant, Type, Variable, Predicate, Atom, Clause, c_const, \
    construct_recursive_rule
from ..lpsolver import LPSolver


class MiniKanren(LPSolver):

    def __init__(self, knowledge_base=None, background_knowledge=None):
        super().__init__(KANREN_LOGPY, knowledge_base, background_knowledge)

    def declare_constant(self, elem_constant: Constant) -> None:
        elem_constant.add_engine_object(elem_constant.name)

    def declare_type(self, elem_type: Type) -> None:
        # No types in miniKanren
        pass

    def declare_variable(self, elem_variable: Variable) -> None:
        v = kanren.var()
        elem_variable.add_engine_object(v)

    def declare_predicate(self, elem_predicate: Predicate) -> None:
        # predicate declared once facts and rules are added
        #     predicates used as facts need to be declared as
        pass

    def assert_fact(self, fact: Atom) -> None:
        try:
            fact.get_predicate().get_engine_obj(KANREN_LOGPY)
        except Exception:
            fact.get_predicate().add_engine_object((KANREN_LOGPY, kanren.Relation()))

        kanren.fact(fact.get_predicate().get_engine_obj(KANREN_LOGPY),
                    *[x.as_kanren() for x in fact.get_terms()]
                    )

    def assert_rule(self, rule: Union[Clause, Sequence[Clause]]) -> None:
        # only needs to add a miniKanren object to the predicate in the head
        if isinstance(rule, Clause):
            if rule.is_recursive():
                raise Exception(f"recursive rule needs to be added together with the base base: {rule}")
            else:
                obj = rule.as_kanren()
                rule.get_head().get_predicate().add_engine_object((KANREN_LOGPY, obj))
        else:

            obj = construct_recursive_rule(rule)

            rule[0].get_head().get_predicate().add_engine_object((KANREN_LOGPY, obj))

    def _query(self, query: Union[Atom, Clause], num_sols=1) -> Tuple[Sequence[Sequence[str]], Sequence[Variable]]:
        if isinstance(query, Atom):
            vars = [x.as_kanren() for x in query.get_variables()]
            ori_vars = [x for x in query.get_variables()]
            if len(vars) == 0:
                # needed in case
                ori_vars = [x.as_kanren() for x in query.get_terms()]
        else:
            vars = [x.as_kanren() for x in query.get_head().get_variables()]
            ori_vars = [x for x in query.get_head().get_variables()]

        if isinstance(query, Atom):
            goals = [query.as_kanren()]
        else:
            goals = [x.as_kanren() for x in query.get_atoms()]

        return kanren.run(num_sols, vars, *goals), ori_vars

    def has_solution(self, query: Union[Atom, Clause]) -> bool:
        if isinstance(query, (Atom, Clause)):
            res, _ = self._query(query, num_sols=1)

            return True if res else False
        else:
            raise Exception(f"cannot query {type(query)}")

    def one_solution(self, query: Union[Atom, Clause]) -> Dict[Variable, Constant]:
        res, vars = self._query(query, num_sols=1)

        if len(res) == 0:
            return {}

        return dict(zip(vars, [c_const(x, vars[ind].get_type()) for ind, x in enumerate(res[0])]))

    def all_solutions(self, query: Union[Atom, Clause]) -> Sequence[Dict[Variable, Constant]]:
        res, vars = self._query(query, num_sols=0)

        if len(res) == 0:
            return []

        return [
            dict(zip(vars, [c_const(y, vars[ind].get_type()) for ind, y in enumerate(x)])) for x in res
        ]