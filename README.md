# loreleai
Lorelai aims to be Keras for **Lo**gical **re**asoning and **le**arning in AI.
It provides a unified language for expressing logical theories and connects it to various backends (Prolog, Answer Set Programming, Datalog, ...) to reason with the provided theories.


# Quick start
Loreleai allows you to easy specify you knowledge and ask queries about it

```python
from loreleai.language.lp import c_const, c_var, c_pred
from loreleai.reasoning.lp.datalog import MuZ

p1 = c_const("p1")                      # create a constant with the name 'p1'
p2 = c_const("p2")
p3 = c_const("p3")

parent = c_pred("parent", 2)            # create a predicate/relation 'parent'
grandparent = c_pred("grandparent", 2)

f1 = parent(p1, p2)                     # create the fact 'parent(p1, p2)'
f2 = parent(p2, p3)

V1 = c_var("X")                         # create a variable named 'X'
V2 = c_var("Y")
V3 = c_var("Z")

# create a clause defining the grandparent relation
cl = (grandparent(V1, V3) <= parent(V1, V2) & parent(V2, V3))

solver = MuZ()                          # instantiate the solver
                                        # Z3 datalog (muZ)
solver.assert_fact(f1)                  # assert a fact
solver.assert_fact(f2)
solver.assert_rule(cl)                  # assert a rule

solver.has_solution(grandparent(p1, p3))# ask whether there is a solution to a query
solver.all_solutions(parent(V1, V2))    # ask for all solutions
solver.one_solution(grandparent(p1, V1))# ask for a single solution

```

# Supported reasoning engines

## Prolog

Currently supported:
 - none yet
 
Considering:
 - primitive [SWI Prolog](https://www.swi-prolog.org/) commandline interface
 - [Pyrolog](https://bitbucket.org/cfbolz/pyrolog/src/default/)
 - [sPyrolog](https://github.com/leonweber/spyrolog)  (makes Pyrolog redundant?)
 - an actual SWI Prolog wrapper (to be made from scratch)
 

## Relational programming
Prolog without side-effects (cut and so on)

Currently supported:
 - [miniKanren](https://github.com/pythological/kanren); seems to be actively maintained
 
Considering:
 - [logpy](https://github.com/logpy/logpy) (pre-decessor of miniKanren?)
 - [microkanren](https://github.com/ethframe/microkanren)
 - [microkanrenpy](https://microkanrenpy.readthedocs.io/en/latest/index.html)
 
## Datalog
A subset of Prolog without functors/structures

Currently supported:
 - [muZ (Z3's datalog engine)](http://www.cs.tau.ac.il/~msagiv/courses/asv/z3py/fixedpoints-examples.htm)
 
Considering:
 - [pyDatalog](https://sites.google.com/site/pydatalog/home)
 
## Deductive databases

Currently supported:
 - none yet
 
Considering:
 - [Grakn](https://grakn.ai/)


# Roadmap

## First direction: reasoning engines


## Second directions: learning primitives

Other features:
 - ILP learners
 - 



# Requirements

  - pyswip
  - problog
  - ortools
  - minikanren
  - z3-solver
  - black
  
# Notes for different engines

## SWI Prolog
For using SWI prolog, check the install instructions: https://github.com/yuce/pyswip/blob/master/INSTALL.md

## Z3

Z3Py scripts stored in arbitrary directories can be executed if the 'build/python' directory is added to the PYTHONPATH environment variable and the 'build' directory is added to the DYLD_LIBRARY_PATH environment variable.
