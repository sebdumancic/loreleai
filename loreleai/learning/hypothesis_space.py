import typing
from abc import ABC, abstractmethod

import networkx as nx

from loreleai.language.commons import FillerPredicate
from loreleai.language.lp import Predicate, Clause, Procedure, Atom, Variable
from loreleai.learning.language_manipulation import plain_extension


class HypothesisSpace(ABC):
    def __init__(
            self,
            primitives: typing.Union[Predicate, typing.Dict],
            head_constructor: typing.Union[Predicate, FillerPredicate],
            discard_singletons: bool = True,
            connected_clauses: bool = True,
            connected_bodies: bool = False,
            only_recursive_procedures: bool = True,
    ) -> None:
        self._primitives = primitives
        self._primitives = self._prepare_primitives()
        self._head_constructor: typing.Union[Predicate, FillerPredicate] = head_constructor
        # Predicate -> use this predicate in the head
        # FillerPredicate -> create new head predicate for each clause/procedure
        self._hypothesis_space = None
        self._discard_singletons = discard_singletons
        self._connected_clauses = connected_clauses
        self._connected_bodies = connected_bodies
        self._only_recursions = only_recursive_procedures
        self._pointers: typing.Dict[str, typing.Union[Clause, Procedure]] = {"main": None}

    def _prepare_primitives(self):
        new_primitives = {}

        for prim in self._primitives:
            if isinstance(prim, dict):
                key = list(prim.keys())
                if len(key) > 1 or not isinstance(key[0], Predicate):
                    raise Exception(f"don't know how to turn {prim} into extension primitive")

                new_primitives[key[0]] = prim[key[0]]
            elif isinstance(prim, Predicate):
                def my_func(clause):
                    return plain_extension(clause, prim, connected_clauses=self._connected_clauses)

                new_primitives[prim] = my_func
            else:
                raise Exception(f"don't know how to turn {prim} into extension primitive")

        return new_primitives

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
    def get_current_candidate(self) -> typing.Union[Clause, Procedure]:
        """
        Get the next candidate
        """
        raise NotImplementedError()

    @abstractmethod
    def get_successors_of(self, node: typing.Union[Clause, Procedure]) -> typing.Sequence[typing.Union[Clause, Procedure]]:
        """
        Get all successors of the provided node
        """
        raise NotImplementedError()

    @abstractmethod
    def get_predecessor_of(self, node: typing.Union[Clause, Procedure]) -> typing.Union[Clause, Procedure]:
        """
        Returns the predecessor of the node
        """
        raise NotImplementedError()

    @abstractmethod
    def move_pointer_to(self, node: typing.Union[Clause, Procedure], pointer_name: str = 'main') -> None:
        """
        Moves the current candidate pointer to
        """
        raise NotImplementedError()


class RecursiveComposition:

    def __init__(self, recursive_clause: Clause, pointer_name: str):
        self._recursive_case: Clause = recursive_clause
        self._pointer_name: str = pointer_name

    def _next_bfs(self, hypothesis_space: HypothesisSpace):
        """
        Constructs the next recursion in a bread-first manner
        """
        pass

    def _next_dfs(self,  hypothesis_space: HypothesisSpace):
        """
        Constructs the next recursion in a depth-first manner
        """
        pass

    def next(self, hypothesis_space: HypothesisSpace, method: str = 'bfs'):
        assert method in ['bfs', 'dfs']

        if method == 'bfs':
            return self._next_bfs(hypothesis_space)
        else:
            return self._next_dfs(hypothesis_space)


class TopDownHypothesisSpace(HypothesisSpace):
    def __init__(
        self,
        primitives: typing.Union[Predicate, typing.Dict],
        head_constructor: typing.Union[Predicate, FillerPredicate],
        discard_singletons: bool = True,
        connected_clauses: bool = True,
        only_recursive_procedures: bool = True
    ):
        super().__init__(
            primitives,
            head_constructor,
            discard_singletons=discard_singletons,
            only_recursive_procedures=only_recursive_procedures,
            connected_clauses=connected_clauses,
        )
        self._hypothesis_space = nx.DiGraph()
        self._root_node: typing.Union[Clause, Procedure] = None
        self._invented_predicate_count = 0

    def initialise(self) -> None:
        """
        Initialises the search space
        """
        if isinstance(self._head_constructor, Predicate):
            head_variables = [Variable(chr(x)) for x in range(ord("A"), ord("B"))][
                : self._head_constructor.get_arity()
            ]
            head: Atom = Atom(self._head_constructor, head_variables)
            clause = Clause(head, [])
            self._hypothesis_space.add_node(clause)
            self._pointers['main'] = clause
            self._root_node = clause
        elif isinstance(self._head_constructor, str):
            if self._head_constructor.startswith("*"):
                raise Exception("not supported yet (head constructor *)")
            elif self._head_constructor.startswith("?"):
                raise Exception("not supported yet (head constructor ?)")
        else:
            raise Exception(f"Unknown head constructor ({self._head_constructor}")

    def _insert_node(self, node: typing.Union[Clause, Procedure]) -> None:
        """
        Inserts a clause/procedure into the hypothesis space
        """
        self._hypothesis_space.add_node(node, active=True, ignored=False, visited=False, last_visited_from=None)

    def _insert_edge(self, parent: typing.Union[Clause, Procedure], child: typing.Union[Clause, Procedure]) -> None:
        """
        Inserts a directed edge between two clauses/procedures in the hypothesis space
        """
        self._insert_node(child)
        self._hypothesis_space.add_edge(parent, child)

    def register_pointer(self, name: str, init_value: typing.Union[Clause, Procedure] = None):
        """
        Registers a new pointer. If init_value is None, assigns it to the root note
        """
        if name in self._pointers:
            raise Exception(f"pointer {name} already exists!")
        else:
            self._pointers[name] = self._root_node if init_value is None else init_value

    def _expand_with_given_head_predicate(self, node: typing.Union[Clause, Procedure]) -> typing.Sequence[typing.Union[Clause, Procedure]]:
        """
                Expands the provided node with provided primitives (extensions become its children in a graph)
                when head_constructor is set to a Predicate

                returns the expanded constructs
        """
        expansions = []

        for prim_ind in range(len(self._primitives)):
            prim = self._primitives[prim_ind]
            if isinstance(prim, list):
                for item in range(len(prim)):
                    expansions += prim[item](node)
            else:
                expansions += prim(node)

        for ind in range(len(expansions)):
            self._insert_edge(node, expansions[ind])

        return expansions

    def _expand_with_filler_predicate(self, node: typing.Union[Clause, Procedure]) -> typing.Sequence[typing.Union[Clause, Procedure]]:
        """
        Expands the provided node when head_constructor is set to '?'
        """
        expansions = []

        for prim_ind in range(len(self._primitives)):
            prim = self._primitives[prim_ind]
            if isinstance(prim, list):
                for item in range(len(prim)):
                    exp = prim[item](node)
                    head_pred = node.get_head().get_predicate()
                    expansions += [x.substitute_predicate(head_pred, self._head_constructor.new()) for x in exp]
            else:
                exp = prim(node)
                head_pred = node.get_head().get_predicate()
                expansions += [x.substitute_predicate(head_pred, self._head_constructor.new()) for x in exp]

        for ind in range(len(expansions)):
            self._insert_edge(node, expansions[ind])

        return expansions

    def expand(self, node: typing.Union[Clause, Procedure]) -> typing.Sequence[typing.Union[Clause, Procedure]]:
        """
        Expands the provided node with provided primitives (extensions become its children in a graph)

        returns the expanded constructs
        """
        if isinstance(self._head_constructor, Predicate):
            return self._expand_with_given_head_predicate(node)

    def block(self, node: typing.Union[Clause, Procedure]) -> None:
        """
        Removes the node from the hypothesis space (and all of its descendents)
        """

        children = self._hypothesis_space.successors(node)
        self._hypothesis_space.remove_node(node)

        if len(children) > 0:
            for ch_ind in range(len(children)):
                self.block(children[ch_ind])

    def ignore(self, node: typing.Union[Clause, Procedure]) -> None:
        """
        Sets the node to be ignored. That is, the node will be expanded, but not taken into account as a candidate
        """
        self._hypothesis_space.nodes[node]['ignored'] = True

    def get_current_candidate(self, name: str = 'main') -> typing.Union[Clause, Procedure]:
        """
        Get the current program candidate (the current pointer)
        """
        return self._pointers[name]

    def move_pointer_to(self, node: typing.Union[Clause, Procedure], pointer_name: str = 'main') -> None:
        """
        Moves the pointer to the pre-defined node
        """
        self._hypothesis_space.nodes[node]['last_visited_from'] = self._pointers[pointer_name]
        self._pointers[pointer_name] = node

    def get_predecessor_of(self, node: typing.Union[Clause, Procedure]) -> typing.Union[Clause, Procedure]:
        return self._hypothesis_space.nodes[node]['last_visited_from']

    def get_successors_of(self, node: typing.Union[Clause, Procedure]) -> typing.Sequence[typing.Union[Clause, Procedure]]:
        return list(self._hypothesis_space.successors(node))






