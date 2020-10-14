
import os
from functools import reduce
from typing import Union

import pyxsb

from loreleai.language.lp import Variable, Structure, List, Atom, Clause, c_var, c_symbol
from .Prolog import Prolog


def _is_list(term: str):
    return term.startswith('[')


def _is_structure(term: str):
    first_bracket = term.find('(')

    if first_bracket == -1:
        return False
    else:
        tmp = term[:first_bracket]
        return all([x.isalnum() for x in tmp]) and tmp[0].isalpha()


def _pyxsb_string_to_const_or_var(term: str):
    if term[0].islower():
        return c_symbol(term)
    elif term.isnumeric():
        if '.' in term:
            return float(term)
        else:
            return int(term)
    else:
        return c_var(term)


def _extract_arguments_from_compound(term: str):
    if _is_list(term):
        term = term[1:-1]  # remove '[' and ']'
    else:
        first_bracket = term.find('(')
        term = term[first_bracket+1:-1] # remove outer brackets

    args = []
    open_brackets = 0
    last_open_char = 0
    for i in range(len(term)):
        char = term[i]
        if term[i] in ['(', '[']:
            open_brackets += 1
        elif term[i] in [')', ']']:
            open_brackets -= 1
        elif term[i] == ',' and open_brackets == 0:
            args.append(term[last_open_char:i])
            last_open_char = i + 1
        elif i == len(term) - 1:
            args.append(term[last_open_char:])
        else:
            pass

    return args


def _pyxsb_string_to_structure(term: str):
    first_bracket = term.find('(')
    functor = term[:first_bracket]
    args = [_pyxsb_string_to_pylo(x) for x in _extract_arguments_from_compound(term)]
    functor = c_symbol(functor, arity=len(args))

    return Structure(functor, args)


def _pyxsb_string_to_list(term: str):
    args = [_pyxsb_string_to_pylo(x) for x in _extract_arguments_from_compound(term)]
    return List(args)


def _pyxsb_string_to_pylo(term: str):
    if _is_list(term):
        return _pyxsb_string_to_list(term)
    elif _is_structure(term):
        return _pyxsb_string_to_structure(term)
    else:
        return _pyxsb_string_to_const_or_var(term)


class XSBProlog(Prolog):

    def __init__(self, exec_path=None):
        if exec_path is None:
            exec_path = os.getenv('XSB_HOME', None)
            raise Exception(f"Cannot find XSB_HOME environment variable")
        pyxsb.pyxsb_init_string(exec_path)
        super().__init__("XSBProlog")

    def __del__(self):
        pyxsb.pyxsb_close()

    def consult(self, filename: str):
        return pyxsb.pyxsb_command_string(f"consult('{filename}').")

    def use_module(self, module: str, **kwargs):
        assert 'predicates' in kwargs, "XSB Prolog: need to specify which predicates to import from module"
        predicates = kwargs['predicates']
        command = f"use_module({module},[{','.join([x.get_name() + '/' + str(x.get_arity()) for x in predicates])}])."
        return pyxsb.pyxsb_command_string(command)

    def asserta(self, clause: Union[Clause, Atom]):
        if isinstance(clause, Atom):
            return pyxsb.pyxsb_command_string(f"asserta({clause}).")
        else:
            return pyxsb.pyxsb_command_string(f"asserta(({clause})).")

    def assertz(self, clause: Union[Atom, Clause]):
        if isinstance(clause, Atom):
            return pyxsb.pyxsb_command_string(f"assertz({clause}).")
        else:
            return pyxsb.pyxsb_command_string(f"assertz(({clause})).")

    def retract(self, clause: Union[Atom, Clause]):
        if isinstance(clause, Atom):
            return pyxsb.pyxsb_command_string(f"retract({clause}).")
        else:
            return pyxsb.pyxsb_command_string(f"retract(({clause})).")

    def has_solution(self, *query):
        string_repr = ','.join([str(x) for x in query])
        res = pyxsb.pyxsb_query_string(f"{string_repr}.")

        if res:
            pyxsb.pyxsb_close_query()

        return True if res else False

    def query(self, *query, **kwargs):
        if 'max_solutions' in kwargs:
            max_solutions = kwargs['max_solutions']
        else:
            max_solutions = -1

        vars_of_interest = [[y for y in x.get_arguments() if isinstance(y, Variable)] for x in query]
        vars_of_interest = reduce(lambda x, y: x + y, vars_of_interest, [])
        vars_of_interest = reduce(lambda x, y: x + [y] if y not in x else x, vars_of_interest, [])

        string_repr = ','.join([str(x) for x in query])
        res = pyxsb.pyxsb_query_string(f"{string_repr}.")

        all_solutions = []
        while res and max_solutions != 0:
            vals = [x for x in res.strip().split(";")]
            var_assignments = [_pyxsb_string_to_pylo(x) for x in vals]
            all_solutions.append(dict([(v, s) for v, s in zip(vars_of_interest, var_assignments)]))

            res = pyxsb.pyxsb_next_string()
            max_solutions -= 1

        return all_solutions

    def register_foreign(self, pyfunction, arity):
        raise Exception("support for foreign predicates not supported yet")

