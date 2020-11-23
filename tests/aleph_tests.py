import datetime

from loreleai.learning.learners import Aleph
from loreleai.learning.task import Task,Knowledge
from loreleai.language.commons import c_type,Clause, Term, Variable, Constant, c_pred,c_const,c_var, Body
from loreleai.learning import HypothesisSpace
from loreleai.reasoning.lp.prolog import SWIProlog
from loreleai.reasoning.lp.datalog import MuZ
from loreleai.learning.eval_functions import Coverage, Compression
from loreleai.learning.language_manipulation import plain_extension, variable_instantiation
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_duplicated_literal, has_singleton_vars

def instant():
    mother = c_pred("mother",2)
    father = c_pred("father",2)
    grandparent = c_pred("grandparent",2)
    tom = c_const("tom")
    john  = c_const("john")
    x = c_var("X")
    y = c_var("Y")

    extensions = [lambda x,y=pred: plain_extension(x,y,connected_clauses=True) for pred in [mother,father]]
    instantiations = [lambda x,y=const: variable_instantiation(x,y) for const in [tom,john]]

    hs = TopDownHypothesisSpace(primitives=extensions+instantiations,
                                head_constructor=grandparent,
                                expansion_hooks_reject=[lambda x, y: has_singleton_vars(x, y),
                                                        lambda x, y: has_duplicated_literal(x, y)])

    b = Body(mother(x,tom),father(x,john))
    print("Heads: {}".format(hs._create_possible_heads(b)))

    cands = hs.get_current_candidate()
    while len(cands) > 0:
        for i in range(len(cands[:30])):
            print("{}: {}".format(i,cands[i]))
        x = int(input("Clause to expand: "))
        print("Expanding clause {}".format(cands[x]))
        cands = hs.expand(cands[x])

def learn_with_constants():
    """
    Consider a row of blocks [ block1 block2 block3 block4 block5 block6 ]
    The order of this row is expressed using follows(X,Y)
    The color of a block is expressed using color(X,Color)
    
    Goal: learn a function f that says: a block is positive when it is followed by a red block
    pos(X) :- next(X,Y), color(Y,red)
    """
    block = c_type("block")
    col = c_type("col")

    block1 = c_const("block1",domain=block) # blue -> positive
    block2 = c_const("block2",domain=block) # red
    block3 = c_const("block3",domain=block) # green -> positive
    block4 = c_const("block4",domain=block) # red -> positive
    block5 = c_const("block5",domain=block) # red
    block6 = c_const("block6",domain=block) # green
    block7 = c_const("block7",domain=block) # blue
    block8 = c_const("block8",domain=block) # blue

    red = c_const("red",domain="col")
    green = c_const("green",domain="col")
    blue = c_const("blue",domain="col")

    follows = c_pred("next",2,domains=[block,block])
    color = c_pred("color",2,domains=[block,col])
    
    # Predicate to learn:
    f = c_pred("f",1,domains=[block])

    bk = Knowledge(
        follows(block1,block2), follows(block2,block3), follows(block3,block4),
        follows(block4,block5), follows(block5,block6), follows(block6,block7),
        follows(block7,block8), color(block1,blue), color(block2, red),
        color(block3,green), color(block4,red), color(block5,red),
        color(block6,green), color(block7,blue), color(block8,blue)
    )

    pos = {f(x) for x in [block1,block3,block4]}
    neg = {f(x) for x in [block2,block5,block6,block7,block8]}

    task = Task(positive_examples=pos, negative_examples=neg)
    solver = SWIProlog()

    # EvalFn must return an upper bound on quality to prune search space.
    eval_fn = Coverage(return_upperbound=True)
    # eval_fn2 = Compression(return_upperbound=True)
    # eval_fn3 = Compression(return_upperbound=True)

    learner = Aleph(solver,eval_fn,max_body_literals=5,do_print=False)
    # learner2 = Aleph(solver,eval_fn2,max_body_literals=4,do_print=True)
    # learner3 = Aleph(solver,eval_fn3,max_body_literals=4,do_print=True)


    start = datetime.datetime.now()
    prog = learner.learn(task,bk,None)
    end = datetime.datetime.now()
    print("Final program: {}".format(str(prog)))
    print("Learning took {}s, and {} clauses were considered.".format((end-start).total_seconds(),learner._eval_fn._clauses_evaluated))
    


def learn_simpsons():    
    # define the predicates
    father = c_pred("father", 2)
    mother = c_pred("mother", 2)
    grandparent = c_pred("grandparent", 2)

    # specify the background knowledge
    background = Knowledge(
        father("homer", "bart"), father("homer", "lisa"), father("homer", "maggie"), 
        mother("marge", "bart"), mother("marge", "lisa"),mother("marge","maggie"),
        mother("mona","homer"),father("abe","homer"),
        mother("jacqueline","marge"),father("clancy","marge")
    )

    # positive examples
    pos = {
        grandparent("abe", "bart"), 
        grandparent("abe", "lisa"), 
        grandparent("abe", "maggie"), 
        grandparent("mona", "bart"), 
        grandparent("abe", "lisa"), 
        grandparent("abe", "maggie"), 
        grandparent("jacqueline", "bart"), 
        grandparent("jacqueline", "lisa"), 
        grandparent("jacqueline", "maggie"), 
        grandparent("clancy", "bart"), 
        grandparent("clancy", "lisa"), 
        grandparent("clancy", "maggie"), 
    }

    # negative examples
    neg = {
        grandparent("abe", "marge"), grandparent("abe", "homer"), grandparent("abe", "clancy"),grandparent("abe","jacqueline"),
        grandparent("homer","marge"), grandparent("homer","jacqueline"),grandparent("jacqueline","marge"),
        grandparent("clancy","homer"),grandparent("clancy","abe")
    }

    task = Task(positive_examples=pos, negative_examples=neg)
    solver = SWIProlog()

    # EvalFn must return an upper bound on quality to prune search space.
    eval_fn = Coverage(return_upperbound=True)
    eval_fn2 = Compression(return_upperbound=True)
    eval_fn3 = Compression(return_upperbound=True)

    learner = Aleph(solver,eval_fn,max_body_literals=4,do_print=True)
    learner2 = Aleph(solver,eval_fn2,max_body_literals=4,do_print=True)
    learner3 = Aleph(solver,eval_fn3,max_body_literals=4,do_print=True)

    start = datetime.datetime.now()
    pr = learner.learn(task,background,None)
    end = datetime.datetime.now()
    print("Final program: {}".format(str(pr)))
    print("Learning took {}s, and {} clauses were considered.".format((end-start).total_seconds(),learner._eval_fn._clauses_evaluated))

    # pr = learner2.learn(task,background,None)
    # print("Final program: {}".format(str(pr)))

    # pr = learner3.learn(task,background,None)
    # print("Final program: {}".format(str(pr)))


if __name__ == "__main__":
    # instant()
    # learn_simpsons()
    learn_with_constants()





    
    



