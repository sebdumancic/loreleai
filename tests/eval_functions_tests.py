import typing
from abc import ABC, abstractmethod
import math

from orderedset import OrderedSet

from loreleai.language.lp import c_pred, Clause, Procedure, Atom, Variable
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_singleton_vars, has_duplicated_literal
from loreleai.learning.language_manipulation import plain_extension
from loreleai.learning.task import Task, Knowledge
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog
from loreleai.learning.learners import SimpleBreadthFirstLearner
from loreleai.learning.eval_functions import Accuracy, Compression, Coverage, Entropy

# define the predicates
father = c_pred("father", 2)
mother = c_pred("mother", 2)
grandparent = c_pred("grandparent", 2)
# specify the background knowledge
background = Knowledge(father("a", "b"), mother("a", "b"), mother("b", "c"),
                       father("e", "f"), father("f", "g"),
                       mother("h", "i"), mother("i", "j"))

# positive examples
pos = {grandparent("a", "c"), grandparent("e", "g"), grandparent("h", "j")}

# negative examples
neg = {grandparent("a", "b"), grandparent("a", "g"), grandparent("i", "j")}
task = Task(positive_examples=pos, negative_examples=neg)


# 1. A clause of length 3 covers all positive examples and no negative examples
covered = pos
x = Variable("X")
y = Variable("Y")
# contents of clause don't matter, only length
cl = grandparent(x,y) <= father(x,y) & father(x,y) & father(x,y)

# accuracy = P/(N+P)
acc = Accuracy()
# coverage = P-N
cov = Coverage()
# compression = P-N-L+1
comp = Compression()
# entropy = p log p + (1-p) log (1-p) where p = P/(P + N)
entr = Entropy()

assert(acc.evaluate(cl,task,covered) == 1)
assert(cov.evaluate(cl,task,covered) == 3)
assert(comp.evaluate(cl,task,covered) == 3-3+1)
assert(entr.evaluate(cl,task,covered) == 0)

# 2. A clause of length 2 covers 3 positive examples, 2 negative examples
covered = list(pos) + [grandparent("a", "b"),grandparent("a", "g")]
# contents of clause don't matter, only length
cl = grandparent(x,y) <= father(x,y) & father(x,y)

assert(acc.evaluate(cl,task,covered) == 3/5)
assert(cov.evaluate(cl,task,covered) == 3-2)
assert(comp.evaluate(cl,task,covered) == 3-2-2+1)
assert(entr.evaluate(cl,task,covered) == 0.29228525323862886)

# 3. A clause of length 4 covers 0 positive examples, 3 negative examples
covered = neg
# contents of clause don't matter, only length
cl = grandparent(x,y) <= father(x,y) & father(x,y) & father(x,y) & father(x,y)

assert(acc.evaluate(cl,task,covered) == 0/3)
assert(cov.evaluate(cl,task,covered) == 0-3)
assert(comp.evaluate(cl,task,covered) == -3-4+1)
assert(entr.evaluate(cl,task,covered) == 0)


