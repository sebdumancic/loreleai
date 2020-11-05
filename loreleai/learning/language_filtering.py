from functools import reduce

from loreleai.language.commons import Body, Atom, Not, Predicate
from .utilities import are_variables_connected

"""
It contains the functions used to prune the search space
"""


def has_singleton_vars(head: Atom, body: Body) -> bool:
    """
    Returns True is the clause has a singleton variable (appears only once)
    """
    if len(body) == 0:
        return False

    vars = {}
    head_vars = head.get_variables()
    for ind in range(len(head_vars)):
        if head_vars[ind] not in vars:
            vars[head_vars[ind]] = head_vars.count(head_vars[ind])

    bvars = body.get_variables()
    body_vars_flat = reduce(lambda x, y: x + y, [x.get_variables() for x in body.get_literals()], [])
    for ind in range(len(bvars)):
        if bvars[ind] in vars:
            vars[bvars[ind]] += body_vars_flat.count(bvars[ind])
        else:
            vars[bvars[ind]] = body_vars_flat.count(bvars[ind])

    return True if any([k for k, v in vars.items() if v == 1]) else False


def max_var(head: Atom, body: Body, max_count: int) -> bool:
    """
    Return True if there are no more than max_count variables in the clause
    """
    vars = body.get_variables()
    for v in head.get_variables():
        if v not in vars:
            vars += [v]
    return True if len(vars) <= max_count else False


def connected_body(head: Atom, body: Body) -> bool:
    """
    Returns True if variables in the body cannot be partitioned in two non-overlapping sets
    """
    if len(body) == 0:
        return True
    return are_variables_connected([x.get_atom() if isinstance(x, Not) else x for x in body.get_literals()])


def connected_clause(head: Atom, body: Body) -> bool:
    """
    Returns True is the variables in the clause cannot be partitioned in two non-overlapping sets
    """
    if len(body) == 0:
        return True
    return are_variables_connected([x.get_atom() if isinstance(x, Not) else x for x in body.get_literals() + [head]])


def negation_at_the_end(head: Atom, body: Body) -> bool:
    """
    Returns True is negations appear after all positive literals
    """
    pos_location = -1
    neg_location = -1
    lits = body.get_literals()

    for ind in range(len(lits)):
        if isinstance(lits[ind], Atom):
            pos_location = ind
        elif neg_location < 0:
            neg_location = ind

    return False if (-1 < neg_location < pos_location) else True


def max_pred_occurrences(head: Atom, body: Body, pred: Predicate, max_occurrence: int) -> bool:
    """
    Returns True if the predicate pred does not appear more than max_occurrence times in the clause
    """
    preds = [x for x in body.get_literals() if x.get_predicate() == pred]

    return len(preds) <= max_occurrence


def has_duplicated_literal(head: Atom, body: Body) -> bool:
    """
    Returns True if there are duplicated literals in the body
    """
    return len(body) != len(set(body.get_literals()))




