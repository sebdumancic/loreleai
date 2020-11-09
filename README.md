# loreleai
Lorelai aims to be Keras for **Lo**gical **re**asoning and **le**arning in AI.
It provides a unified language for expressing logical theories and connects it to various backends (Prolog, Answer Set Programming, Datalog, ...) to reason with the provided theories.

# Installation

`loreleai` depends on [pylo](https://github.com/sebdumancic/pylo2) to interface with Prolog engines.
Follow the instructions to install `pylo` [here](https://github.com/sebdumancic/pylo2).

If you will be using a Datalog engine, follow the instructions to install [z3](https://github.com/Z3Prover/z3).

Then clone this repository and run
```shell script
pip install .
```



# Quick start
`loreleai` allows you to easy specify you knowledge and ask queries about it

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
solver.query(parent(V1, V2))    # ask for all solutions
solver.query(grandparent(p1, V1), max_solutions=1)# ask for a single solution
```

Alternatively, `loreleai` provides shortcuts to defining facts
```python
from loreleai.language.lp import  c_pred

parent = c_pred("parent", 2)            # create a predicate/relation 'parent'
grandparent = c_pred("grandparent", 2)

f1 = parent("p1", "p2")                 # 'p1' and 'p2' are automatically parsed into a Constant
f2 = parent("p2", "p3")

query_literal = grandparent("p1", "X")  # 'X' is automatically parsed into a Variable
```

# Supported reasoning engines

### Prolog

Currently supported (via [pylo](https://github.com/sebdumancic/pylo2)):
 - [SWI Prolog](https://www.swi-prolog.org/)
 - [XSB Prolog](http://xsb.sourceforge.net/)
 - [GNU Prolog](http://www.gprolog.org/)
 

### Relational programming
Prolog without side-effects (cut and so on)

Currently supported:
 - [miniKanren](https://github.com/pythological/kanren); seems to be actively maintained
 
 
### Datalog
A subset of Prolog without functors/structures

Currently supported:
 - [muZ (Z3's datalog engine)](http://www.cs.tau.ac.il/~msagiv/courses/asv/z3py/fixedpoints-examples.htm)
 
Considering:
 - [pyDatalog](https://sites.google.com/site/pydatalog/home)
 
### Deductive databases

Currently supported:
 - none yet
 
Considering:
 - [Grakn](https://grakn.ai/)
 
 
### Answer set programming

Currently supported:
  - none yet
  
Considering:
   - [aspirin](https://github.com/potassco/asprin)
   - [clorm](https://github.com/potassco/clorm)
   - [asp-lite](https://github.com/lorenzleutgeb/asp-lite)
   - [hexlite](https://github.com/hexhex/hexlite)
   - [clyngor](https://github.com/aluriak/clyngor)


# Roadmap

### First direction: reasoning engines

 - [x] integrate one solver for each of the representative categories
 - [ ] add support for external predicates (functionality specified in Python)
 - [x] SWI prolog wrapper
 - [ ] include probabilistic engines (Problog, PSL, MLNs)
 - [ ] add parsers for each dialect
 - [ ] different ways of loading data (input language, CSV, ...)
 


### Second directions: learning primitives

 - add learning primitives such as search, hypothesis space generation
 - wrap state of the art learners (ACE, Metagol, Aleph)
 
 
# Code structure

The *language* constructs are in `loreleai/language` folder. 
There is a folder for each dialect of first-order logic.
Currently there are _logic programming_ (`loreleai/language/lp`) and _relational programming_ (`loreleai/language/kanren`).
The implementations of all shared concepts are in `loreleai/language/commons.py` and the idea is to use `__init__.py` files to provide the allowed constructs for each dialect.


The *reasoning* constructs are in `loreleai/reasoning` folder.
The structure is the same as with language. 
Different dialects of logic programming are in the folder `lorelai/reasoning/lp`.


The *learning* primitives are supposed to be in the `loreleai/learning` folder.



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
