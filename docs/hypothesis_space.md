The central design principle in `loreleai` is to explicitly represent the hypothesis space (the space of all programs)  and allow you to manipulate it.
The hypothesis spaces are located in `loreleai.learning.hypothesis_space` and currently implemented ones are: `TopDownHypothesisSpace`.
This document gives a brief introduction how to use them.

# Top down hypothesis space

This is a hypothesis space in which the programs are constructed from the simplest (shortest) to more complicated ones.

## Constructing hypothesis space

To create the hypothesis space, you need to provide the following ingredients:
 - **primitives:** functions that extend a given clause (otherwise known as refinement operators)
 - **head constructor:** instructions how to construct the head of the clauses
 - **expansion hooks:**  functions used to eliminate useless extensions
 

Below is the complete example and this document explains it part by part.
```python
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import connected_clause, has_singleton_vars
from loreleai.learning.language_manipulation import plain_extension
from loreleai.language.lp import c_pred

grandparent = c_pred("grandparent", 2)
father = c_pred("father", 2)
mother = c_pred("mother", 2)

hs = TopDownHypothesisSpace(primitives=[lambda x: plain_extension(x, father, connected_clauses=False),
                                        lambda x: plain_extension(x, mother, connected_clauses=False)],
                            head_constructor=grandparent,
                            expansion_hooks_keep=[lambda x, y: connected_clause(x, y)],
                            expansion_hooks_reject=[lambda x, y: has_singleton_vars(x, y)]
)
```
 
### Primitives

To construct the hypothesis space, you need to provide *primitives* -- functions that take a clause or a body and refine it (add another literal to it).
It accepts it as a list of functions that could be called with a single argument, being `Clause`, `Body` or  `Procedure`.
The function should return a list of extensions of the clause and be type consistent: if an object of the type `Body` is provided, then all extensions should also be of the type `Body` and so on.
`loreleai` provides some primitives, but you can also provide your own functions.

The best way to see how primitive functions should be implemented is to check the implementations `loreleai` provides is `loreleai.learning.language_manipulation`.
Currently, only one such function is implemented: `plain_extension` extends the given body/clause/procedure by adding all possible literals with the provided predicate.
It has the following signature:
```python
def plain_extension(
    clause: typing.Union[Clause, Body, Procedure],
    predicate: Predicate,
    connected_clauses: bool = True,
    negated: bool = False,
) -> typing.Sequence[typing.Union[Clause, Body, Procedure]]
```
where:
 - `clause` is the body/clause/procedure to extend
 - `predicate` is the predicate to add to the body
 - `connected_clauses` is a flag indicating that extensions shoudl result in connected clauses (no disjoint sets of variables)
 - `negated` is a flag indicating that the added literals should be negations
 
 
### Head constructor

Head constructor provides instructions how to construct the heads of clauses.
There are two options to do so:
 - provide the exact predicate to be put in the head
 - provide the `FillerPredicate` (located in `loreleai.learning.utilities`): this means that every unique body will get a new predicate (this is useful for predicate invention)
 
`FillerPredicate` takes several arguments:
 - `prefix` specifies the name of the invented predicates (additionally suffixed by their index)
 - `arity` (optional) of the invented predicates
 - `min_arity` (optional): minimal arity of the invented predicates
 - `max_arity` (optional): maximal arity of the invented predicates
 
`FillerPredicate` can be configured in two ways by:
 - specifying the `arity`: this introduces predicates of fixed arity
 - specifying `min_arity` and `max_arity`: this introduces predicates with arity between `min_arity` to `max_arity`

You have to specify one of the options.


### Expansions hooks

Depending on the primitive functions used, many of the generated clauses will be useless (for example, clauses that are never true in data).
Expansions hooks can be used to eliminate such clauses: every time a hypothesis is expanded, the 'child' hypotheses are checked with expansion hook; if they 'fail' the test, they are eliminated (removed).

Expansion hooks are provided as a list of functions that can be called with a clause as an input.
The functions need to have the following specification:
```python
function_name(head: Atom, body: Body) -> bool
```
where:
 - `head` is the head of a clause
 - `body` is the body of a clause

Expansion hooks come in two flavours:
 - `expansion_hooks_keep`: keeps all expanded hypotheses for which the expansion functions return `True`
 - `expansion_hooks_reject`: rejects all expanded hypothesis for which the expansion functions return `True`
 
`loreleai` implements several of these functions in `loreleai.learning.language_filtering`:
 - `has_singleton_variables(head, body)`: return `True` is a clause has a singleton variable (variable that appears only once)
 - `max_var(head, body, max_count)`: returns `True` if the number of variables in the clause is less or equal to `max_count`
 - `connected_body(head, body)`: returns `True` if the body of the clause is connected (variables cannot be partitioned in disjoint sets)
 - `connected_clause`: returns `True` if the entire clause is connected
 - `negation_at_the_end(head, body)`: returns `True` if negative literals appear after positive literals
 - `max_pred_occurrences(head, body, pred, max_occurrence)`: returns `True` if predicate `pred` appears at most `max_occurence` times in the body of the clause
 - `has_duplicated_literals(head, body):` returns `True` if there are duplicated literals in the body
 
 
### Other options
 - `recursive_procedures`: if set to `True` it will enumerate recursions
 
 
## Using the hypothesis space

The hypothesis space objects offer several methods to manipulate the hypothesis space.

**Important thing to under** is that the unit component in the TopDownHypothesisSpace is `Body`, i.e., the body of the clause.
If the hypothesis space is viewed as a graph, the nodes are bodies and possible heads are kept as 'attributes' of the corresponding node.
This is important to keep in mind because the operations that retrieve candidates from the hypothesis space (e.g., `expand()`) return all candidates with the same body.
Likewise, other operations that manipulate the hypothesis space can affect both a single clause as well as all clauses with the same body.


The `.expand(clause)` method expands/refines the the given clause with all provided primitive functions.
The clause can be either a `Body`, `Clause` or `Procedure` (specifically, `Recursion`).
The method returns all possible expansions of the given clause (not a single one).


The `.block(clause)` method blocks the expansion of the give clause, but keeps the clause itself.
The clause can be either a `Body`, `Clause` or `Procedure` (specifically, `Recursion`).
Every clause with the same body gets blocked.

The `.ignore(clause)` method ignores the clause in the hypothesis space: the clause can be further refined, but will not be returns as a viable candidate.
The clause can be either a `Body`, `Clause` or `Procedure` (specifically, `Recursion`).
If the `Clause` is provided, only that specific clause is ignored.
If the `Body` is provided, every clause with that body is ignored.


The `.remove(clause, remove_entire_body)` method removes the clause from the hypothesis space.
The clause can be of type `Clause`, `Body` or  `Procedure`.
If `Clause` is provided, the specific clause is remove.
If `Body` is provided, every clause with that body is ignore.
If `remove_entire_body` is set to `True`, one can provide a specific clause but all clauses with the same body will be removed.

The `.get_successors_of(clause)` method returns all successors of the clause.
The clause can either be `Clause` or `Body`.
The method returns all clauses that are obtained by extending/refining the body of the given clause.

The `.get_predecessor_of(clause)` method returns all predecessors of the clause.
The clause can either be `Clause` or `Body`.



The hypothesis space contains individual clauses as the basic units.
More complex programs (disjunctions and recursions) can be constructed by combining individual clauses.
Hypothesis space objects in `loreleai` allow you to achieve this through the usage of *pointers*.
A pointer simply holds a position in the hypothesis space.
Multiple pointers can be created and all of them can be moved independently.
Upon construction of the hypothesis space, th `main` pointer is created and assigned to the root node (empty clause).

The `.register_pointer(name, init_value)` method registers new pointer under the name `name`.
If initial value/position of the pointer `init_value` is not provided, it is set to the root node.
`init_value` can be either `Clause` or `Body`; if `Clause` is provided, it is automatically converted to `Body`.

The `.get_current_candidate(pointer_name)` method returns all clauses that can be constructed from the body to which the pointer `pointer_name` is currently assigned to.
If `pointer_name` is not specified, it is assumed that the `main` pointer is in question.
The initial position of the ay pointer is at the root of the TopDownHypothesisSpace.

The `.move_pointer_to(clause, pointer_name)` method moves the pointer `pointer_name` to the body of `clause`.
If `pointer_name` is not specified, it is assumed to be `main` one.
`clause` can be either `Clause` or `Body`; if `Clause` is provided, it is automatically converted to `Body`.

The `.reset_pointer(pointer_name, init_val)` method resets the `pointer_name` pointer to the root or `init_val` if provided.
`init_value` can be either `Clause` or `Body`; if `Clause` is provided, it is automatically converted to `Body`.


Recursions are realised via pointers, if enabled (`recursive_procedures=True`).
Every time an extension/refinement operation results in a recursive clause, a new pointer is created and associated with the hypothesis (every constructed recursive clause is blocked from further expansion).
When `.get_current_candidate` method requests a recursive candidate, the associated pointer traverses the entire hypothesis space in search for valid base cases.
Then it returns all valid recursions.























 


 
 







