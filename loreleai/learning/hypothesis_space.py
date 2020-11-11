import typing
from abc import ABC, abstractmethod
from functools import reduce
from itertools import combinations_with_replacement, combinations

import networkx as nx
from orderedset import OrderedSet

from loreleai.language.lp import (
    Predicate,
    Clause,
    Procedure,
    Atom,
    Body,
    Recursion,
)
from loreleai.learning.utilities import FillerPredicate


class HypothesisSpace(ABC):
    def __init__(
        self,
        primitives: typing.Sequence,
        head_constructor: typing.Union[Predicate, FillerPredicate],
        connected_clauses: bool = True,
        recursive_procedures: bool = False,
        expansion_hooks_keep: typing.Sequence = (),
        expansion_hooks_reject: typing.Sequence = ()
    ) -> None:
        self._primitives: typing.Sequence = primitives
        self._head_constructor: typing.Union[
            Predicate, FillerPredicate
        ] = head_constructor
        # Predicate -> use this predicate in the head
        # FillerPredicate -> create new head predicate for each clause/procedure
        self._hypothesis_space = None
        self._connected_clauses = connected_clauses
        self._use_recursions = recursive_procedures
        self._pointers: typing.Dict[str, Body] = {"main": None}
        self._expansion_hooks_keep = expansion_hooks_keep
        self._expansion_hooks_reject = expansion_hooks_reject

    @abstractmethod
    def initialise(self) -> None:
        """
        Initialise the search space
        """
        raise NotImplementedError()

    @abstractmethod
    def expand(self, node: typing.Union[Clause, Procedure]) -> None:
        """
        Expand the node in the search space
        """
        raise NotImplementedError()

    @abstractmethod
    def block(self, node: typing.Union[Clause, Procedure]) -> None:
        """
        Block expansion of the node
        """
        raise NotImplementedError()

    @abstractmethod
    def ignore(self, node: typing.Union[Clause, Procedure]) -> None:
        """
        Ignores the node, but keeps extending it
        """
        raise NotImplementedError()

    @abstractmethod
    def remove(self, node: typing.Union[Clause, Procedure], remove_entire_body: bool = False,
            not_if_other_parents: bool = True) -> None:
        """
        Removes the clause from the hypothesis space
        """
        raise NotImplementedError()

    @abstractmethod
    def get_current_candidate(self) -> typing.Union[Clause, Procedure]:
        """
        Get the next candidate
        """
        raise NotImplementedError()

    @abstractmethod
    def get_successors_of(
        self, node: typing.Union[Clause, Procedure]
    ) -> typing.Sequence[typing.Union[Clause, Procedure]]:
        """
        Get all successors of the provided node
        """
        raise NotImplementedError()

    @abstractmethod
    def get_predecessor_of(
        self, node: typing.Union[Clause, Procedure]
    ) -> typing.Union[Clause, Procedure]:
        """
        Returns the predecessor of the node
        """
        raise NotImplementedError()

    @abstractmethod
    def move_pointer_to(
        self, node: typing.Union[Clause, Procedure], pointer_name: str = "main"
    ) -> None:
        """
        Moves the current candidate pointer to
        """
        raise NotImplementedError()


class TopDownHypothesisSpace(HypothesisSpace):
    def __init__(
        self,
        primitives: typing.Sequence,
        head_constructor: typing.Union[Predicate, FillerPredicate],
        connected_clauses: bool = True,
        recursive_procedures: bool = False,
        repetitions_in_head_variables: int = 2,
        expansion_hooks_keep: typing.Sequence = (),
        expansion_hooks_reject: typing.Sequence = ()
    ):
        super().__init__(
            primitives,
            head_constructor,
            recursive_procedures=recursive_procedures,
            connected_clauses=connected_clauses,
            expansion_hooks_keep=expansion_hooks_keep,
            expansion_hooks_reject=expansion_hooks_reject
        )
        self._hypothesis_space = nx.DiGraph()
        self._root_node: Body = None
        self._repetition_vars_head = repetitions_in_head_variables
        self._invented_predicate_count = 0
        self._recursive_pointers_count = 0
        self._recursive_pointer_prefix = "rec"
        self.initialise()

    def initialise(self) -> None:
        """
        Initialises the search space
        """
        if isinstance(self._head_constructor, (Predicate, FillerPredicate)):
            if isinstance(self._head_constructor, Predicate):
                # create possible heads
                head_variables = [chr(x) for x in range(ord("A"), ord("Z"))][
                                 : self._head_constructor.get_arity()
                                 ]

                possible_heads = [
                    self._head_constructor(*list(x))
                    for x in combinations_with_replacement(head_variables, self._head_constructor.get_arity())
                ]
            else:
                possible_heads = self._head_constructor.all_possible_atoms()

            # create empty clause
            clause = Body()
            init_head_dict = {"ignored": False, "blocked": False, "visited": False}
            self._hypothesis_space.add_node(clause)
            self._hypothesis_space.nodes[clause]["heads"] = dict([(x, init_head_dict.copy()) for x in possible_heads])
            self._hypothesis_space.nodes[clause]["visited"] = False

            self._pointers["main"] = clause
            self._root_node = clause
        else:
            raise Exception(f"Unknown head constructor ({self._head_constructor}")

    def _create_possible_heads(
        self, body: Body, use_as_head_predicate: Predicate = None
    ) -> typing.Sequence[Atom]:
        """
        Creates possible heads for a given body

        if the _head_constructor is Predicate, it makes all possible combinations that matches the types in the head
        """
        vars = body.get_variables()

        if isinstance(self._head_constructor, Predicate):
            arg_types = self._head_constructor.get_arg_types()

            # matches_vars = []
            # for i in range(len(arg_types)):
            #     matches_vars[i] = []
            #     for var_ind in range(len(vars)):
            #         if arg_types[i] == vars[var_ind].get_type():
            #             matches_vars[i].append(vars[var_ind])
            #
            # bases = [matches_vars[x] for x in range(self._head_constructor.get_arity())]
            # heads = []
            #
            # for comb in product(*bases):
            #     heads.append(Atom(self._head_constructor, list(comb)))
            heads = []
            for comb in combinations(vars, self._head_constructor.get_arity()):
                if [x.get_type() for x in comb] == arg_types:
                    heads.append(Atom(self._head_constructor, list(comb)))
            return heads
        elif isinstance(self._head_constructor, FillerPredicate):
            return self._head_constructor.new_from_body(
                body, use_as_head_predicate=use_as_head_predicate
            )
        else:
            raise Exception(f"Unknown head constructor {self._head_constructor}")

    def _check_if_recursive(self, body: Body):
        """
        checks if the body forms a recursive clause:
            - one of the predicates in the body is equal to the head predicate
            - a predicate constructed by FillerPredicate is in the body
        """
        if isinstance(self._head_constructor, Predicate):
            return True if self._head_constructor is body.get_predicates() else False
        else:
            return (
                True
                if any(
                    [
                        self._head_constructor.is_created_by(x)
                        for x in body.get_predicates()
                    ]
                )
                else False
            )

    def _insert_node(self, node: typing.Union[Body]) -> bool:
        """
        Inserts a clause/procedure into the hypothesis space

        Returns True if successfully inserted (after applying hooks), otherwise returns False
        """
        recursive = self._check_if_recursive(node)

        if recursive and isinstance(self._head_constructor, FillerPredicate):
            recursive_pred = list(
                filter(
                    lambda x: self._head_constructor.is_created_by(x),
                    node.get_predicates(),
                )
            )[0]
            possible_heads = self._create_possible_heads(
                node, use_as_head_predicate=recursive_pred
            )
        else:
            possible_heads = self._create_possible_heads(node)

        # if expansion  hooks available, check if the heads pass
        if self._expansion_hooks_keep:
            possible_heads = [x for x in possible_heads if all([f(x, node) for f in self._expansion_hooks_keep])]

        # if rejection hooks are available, check if the heads fail
        if self._expansion_hooks_reject:
            possible_heads = [x for x in possible_heads if not any([f(x, node) for f in self._expansion_hooks_reject])]

        if possible_heads:
            init_head_dict = {"ignored": False, "blocked": False, "visited": False}
            possible_heads = dict([(x, init_head_dict.copy()) for x in possible_heads])

            self._hypothesis_space.add_node(
                node, last_visited_from=None, heads=possible_heads
            )

            if recursive:
                self._recursive_pointers_count += 1
                pointer_name = (
                    f"{self._recursive_pointer_prefix}{self._recursive_pointers_count}"
                )
                self.register_pointer(pointer_name, self._root_node)
                self._hypothesis_space.nodes[node]["partner"] = pointer_name
                self._hypothesis_space.nodes[node]["blocked"] = True

            return True
        else:
            return False

    def _insert_edge(self, parent: Body, child: Body,) -> None:
        """
        Inserts a directed edge between two clauses/procedures in the hypothesis space
        """
        # if child not in self._hypothesis_space.nodes:
        #     self._insert_node(child)
        self._hypothesis_space.add_edge(parent, child)

    def register_pointer(self, name: str, init_value: Body = None):
        """
        Registers a new pointer. If init_value is None, assigns it to the root note
        """
        if name in self._pointers:
            raise Exception(f"pointer {name} already exists!")
        else:
            self._pointers[name] = self._root_node if init_value is None else init_value

    def reset_pointer(self, name: str = "main", init_value: Body = None):
        """
        Resets the specified pointer to the root or the specified initial value
        """
        self._pointers[name] = self._root_node if init_value is None else init_value

    def _expand_body(self, node: Body) -> typing.Sequence[Body]:
        """
                Expands the provided node with provided primitives (extensions become its children in a graph)

                returns the expanded constructs
        """
        expansions = OrderedSet()

        for item in range(len(self._primitives)):
            exp = self._primitives[item](node)
            expansions = expansions.union(exp)

        # if recursions should be enumerated when FillerPredicate is used to construct the heads
        if isinstance(self._head_constructor, FillerPredicate) and self._use_recursions:
            recursive_cases = self._head_constructor.add_to_body(node)
            for r_ind in range(len(recursive_cases)):
                expansions = expansions.union([node + recursive_cases[r_ind]])

        expansions = list(expansions)

        # add expansions to the hypothesis space
        # if self._insert_node returns False, forget the expansion
        expansions_to_consider = []
        for exp_ind in range(len(expansions)):
            r = self._insert_node(expansions[exp_ind])
            if r:
                expansions_to_consider.append(expansions[exp_ind])

        expansions = expansions_to_consider

        # add edges
        for ind in range(len(expansions)):
            self._insert_edge(node, expansions[ind])

        return expansions

    def retrieve_clauses_from_body(
        self, body: Body
    ) -> typing.Sequence[typing.Union[Clause, Procedure]]:
        """
        Returns all possible clauses given the body
        """
        heads = self._hypothesis_space.nodes[body]["heads"]

        heads = [
            x
            for x in heads
            if not heads[x]["ignored"]
        ]
        return [Clause(x, body) for x in heads]

    def expand(
        self, node: typing.Union[Clause, Procedure]
    ) -> typing.Sequence[typing.Union[Clause, Procedure]]:
        """
        Expands the provided node with provided primitives (extensions become its children in a graph)

        returns the expanded constructs
        if already expanded, returns an empty list
        """
        body = node.get_body()

        if (
            "partner" in self._hypothesis_space.nodes[body]
            or "blocked" in self._hypothesis_space.nodes[body]
        ):
            # do not expand recursions or blocked nodes
            return []

        # check if already expanded
        expansions = list(self._hypothesis_space.successors(body))

        if len(expansions) == 0:
            expansions = self._expand_body(body)
        else:
            return []

        return reduce(
            lambda x, y: x + y,
            [self.retrieve_clauses_from_body(x) for x in expansions],
            [],
        )

    def block(self, node: typing.Union[Clause, Procedure]) -> None:
        """
        Blocks the expansions of the body (but keeps it in the hypothesis space)
        """
        # TODO: make it possible to block only specific clause
        clause = (
            node
            if isinstance(node, Clause)
            else [x for x in node.get_clauses() if x.is_recursive()][0]
        )
        body = clause.get_body()

        self._hypothesis_space.nodes[body]["blocked"] = True

    def remove(
        self, node: typing.Union[Clause, Procedure],
            remove_entire_body: bool = False,
            not_if_other_parents: bool = True
    ) -> None:
        """
        Removes the node from the hypothesis space (and all of its descendents)
        """

        clause = (
            node
            if isinstance(node, Clause)
            else [x for x in node.get_clauses() if x.is_recursive()][0]
        )

        head = clause.get_head()
        body = clause.get_body()

        children = self._hypothesis_space.successors(body)

        if not_if_other_parents:
            # do not remove children that have other parents
            children = [x for x in children if len(self._hypothesis_space.predecessors(x)) <= 1]

        if remove_entire_body:
            # remove entire body
            self._hypothesis_space.remove_node(body)

            for ch_ind in range(len(children)):
                self.remove(Clause(head, body), remove_entire_body=remove_entire_body, not_if_other_parents=not_if_other_parents)
        else:
            # remove just the head
            if head in self._hypothesis_space.nodes[body]["heads"]:
                del self._hypothesis_space.nodes[body]["heads"][head]

            if len(self._hypothesis_space.nodes[body]["heads"]) == 0:
                # if no heads left, remove the entire node
                self._hypothesis_space.remove_node(body)

                for ch_ind in range(len(children)):
                    self.remove(Clause(head, body), remove_entire_body=True)
            else:
                # remove the same head from children
                if len(children) > 0:
                    for ch_ind in range(len(children)):
                        self.remove(Clause(head, children[ch_ind]))

    def ignore(self, node: typing.Union[Clause, Procedure]) -> None:
        """
        Sets the node to be ignored. That is, the node will be expanded, but not taken into account as a candidate
        """
        # TODO: make it possible to ignore the entire body
        clause = (
            node
            if isinstance(node, Clause)
            else [x for x in node.get_clauses() if x.is_recursive()][0]
        )

        head = clause.get_head()
        body = clause.get_body()

        self._hypothesis_space.nodes[body]["heads"][head]["ignored"] = True

    def _get_recursions(self, node: Body) -> typing.Sequence[Recursion]:
        """
        Prepares the valid recursions
        """
        pointer_name = self._hypothesis_space.nodes[node]["partner"]
        init_pointer_value = self._pointers[pointer_name]
        last_pointer_value = None

        valid_heads = list(self._hypothesis_space.nodes[node]["heads"].keys())
        recursions = []

        # for each valid head
        for h_ind in range(len(valid_heads)):
            c_head: Atom = valid_heads[h_ind]
            recursive_clause = Clause(c_head, node)

            frontier = [self._pointers[pointer_name]]

            while len(frontier) > 0:
                focus_node = frontier[0]
                frontier = frontier[1:]

                # find matching heads
                focus_node_heads: typing.Sequence[Atom] = list(
                    self._hypothesis_space.nodes[focus_node]["heads"].keys()
                )
                focus_node_heads = [
                    x
                    for x in focus_node_heads
                    if x.get_predicate().get_arg_types()
                    == c_head.get_predicate().get_arg_types()
                ]

                # prepare recursion
                for bcl_ind in range(len(focus_node_heads)):
                    if isinstance(self._head_constructor, Predicate):
                        recursions.append(
                            Recursion(
                                [
                                    Clause(focus_node_heads[bcl_ind], focus_node),
                                    recursive_clause,
                                ]
                            )
                        )
                    else:
                        # if the filler predicate is used to construct heads, make sure the same head predicate is used
                        head_args = focus_node_heads[bcl_ind].get_arguments()
                        recursions.append(
                            Recursion(
                                [
                                    Clause(
                                        Atom(c_head.get_predicate(), head_args),
                                        focus_node,
                                    ),
                                    recursive_clause,
                                ]
                            )
                        )

                # extend the frontier - exclude recursive nodes
                to_add = [
                    x
                    for x in self._hypothesis_space.successors(focus_node)
                    if "partner" not in self._hypothesis_space.nodes[x]
                ]
                frontier += to_add
                last_pointer_value = focus_node

            # reset the pointer value for next valid head
            self.reset_pointer(pointer_name, init_pointer_value)

        # set the pointer to the last explored clause
        self.reset_pointer(pointer_name, last_pointer_value)

        return recursions

    def get_current_candidate(
        self, name: str = "main"
    ) -> typing.Sequence[typing.Union[Clause, Procedure]]:
        """
        Get the current program candidate (the current pointer)
        """
        if "partner" in self._hypothesis_space.nodes[self._pointers[name]]:
            # recursion
            return self._get_recursions(self._pointers[name])
        else:
            return self.retrieve_clauses_from_body(self._pointers[name])

    def _extract_body(self, clause: typing.Union[Clause, Procedure]) -> Body:
        if isinstance(clause, Clause):
            return clause.get_body()
        elif isinstance(clause, Recursion):
            rec = clause.get_recursive_case()
            if len(rec) == 1:
                return rec[0].get_body()
            else:
                raise Exception(
                    f"got more than one recursive case when extracting the body {clause}"
                )
        else:
            raise Exception(
                f"Don't know how to get a single body from {type(clause)} {clause}"
            )

    def move_pointer_to(
        self, node: typing.Union[Clause, Recursion, Body], pointer_name: str = "main"
    ) -> None:
        """
        Moves the pointer to the pre-defined node
        """
        if isinstance(node, Body):
            body = node
        else:
            body = self._extract_body(node)

        self._hypothesis_space.nodes[body]["last_visited_from"] = self._pointers[
            pointer_name
        ]
        self._pointers[pointer_name] = body

    def get_predecessor_of(
        self, node: typing.Union[Clause, Recursion, Body]
    ) -> typing.Union[Clause, Recursion, Body, typing.Sequence[Clause]]:
        """
        Returns the predecessor of the node = the last position of the pointer before reaching the node
        :param node:
        :return:
        """
        # TODO: make it possible to get all predecessors, not just the last visited from
        if isinstance(node, Body):
            return self._hypothesis_space.nodes[node]["last_visited_from"]
        else:
            if isinstance(node, Clause):
                head = node.get_head()
                body = node.get_body()
            else:
                rec = node.get_recursive_case()
                if len(rec) > 1:
                    raise Exception(
                        "do not support recursions with more than 1 recursive case"
                    )
                else:
                    head = rec[0].get_head()
                    body = rec[0].get_body()

            predecessor = self._hypothesis_space.nodes[body]["last_visited_from"]
            if head in self._hypothesis_space.nodes[predecessor]["heads"]:
                return Clause(head, predecessor)
            else:
                return self.retrieve_clauses_from_body(predecessor)

    def get_successors_of(
        self, node: typing.Union[Clause, Recursion, Body]
    ) -> typing.Sequence[typing.Union[Clause, Body, Procedure]]:
        """
        Returns all successors of the node
        """
        if isinstance(node, Body):
            return list(self._hypothesis_space.successors(node))
        else:
            body = self._extract_body(node)
            return reduce(lambda x, y: x + y, [self.retrieve_clauses_from_body(x) for x in self._hypothesis_space.successors(body)], [])

