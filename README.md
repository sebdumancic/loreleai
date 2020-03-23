# loreleai
Lorelai aims to be Keras for **Lo**gical **re**asoning and **le**arning in AI.
It provides a unified language for expressing logical theories and connects it to various backends (Prolog, Answer Set Programming, Datalog, ...) to reason with the provided theories.


# Requirements

  - pyswip
  - problog
  - ortools
  
# Features  
  
# Roadmap

Currently supported:
 - [muZ (Z3's datalog engine)](http://www.cs.tau.ac.il/~msagiv/courses/asv/z3py/fixedpoints-examples.htm)
 
Working towards the support:
 - primitive [SWI Prolog](https://www.swi-prolog.org/) commandline interface
 - [pyDatalog](https://sites.google.com/site/pydatalog/home)
 - [Pyrolog](https://bitbucket.org/cfbolz/pyrolog/src/default/)
 - [sPyrolog](https://github.com/leonweber/spyrolog)  (makes Pyrolog redundant?)
 - miniKanren: [logpy](https://github.com/logpy/logpy) or [microkanren](https://github.com/ethframe/microkanren); potentially useful resource [link](https://stackoverflow.com/questions/11291242/python-dynamically-create-function-at-runtime)
 - an actual SWI Prolog wrapper (to be made from scratch)
 - [Grakn](https://grakn.ai/)
 
Other features:
 - ILP learners
 - 


  
# Notes for different engines

## SWI Prolog
For using SWI prolog, check the install instructions: https://github.com/yuce/pyswip/blob/master/INSTALL.md

## Z3

Z3Py scripts stored in arbitrary directories can be executed if the 'build/python' directory is added to the PYTHONPATH environment variable and the 'build' directory is added to the DYLD_LIBRARY_PATH environment variable.
