import datetime

from loreleai.learning.learners import Aleph
from loreleai.learning.task import Task,Knowledge
from loreleai.language.lp import c_type,Clause, Term, Variable, Constant, c_pred,c_const,c_var, Body
from loreleai.learning import HypothesisSpace
from loreleai.reasoning.lp.prolog import SWIProlog
from loreleai.reasoning.lp.datalog import MuZ
from loreleai.learning.eval_functions import Coverage, Compression, Accuracy
from loreleai.learning.language_manipulation import plain_extension, variable_instantiation
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_duplicated_literal, has_singleton_vars


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

    follows = c_pred("follows",2,domains=[block,block])
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
    eval_fn1 = Coverage(return_upperbound=True)
    eval_fn2 = Compression(return_upperbound=True)
    eval_fn3 = Accuracy(return_upperbound=True)

    learners = [Aleph(solver,eval_fn,max_body_literals=4,do_print=False) 
                    for eval_fn in [eval_fn1,eval_fn3]]

    for learner in learners:
        res = learner.learn(task,bk,None,minimum_freq=1)
        print(res)
    

def learn_text():
    """
    We describe piece of text spanning multiple lines:
    "node A red <newline> node B green <newline> node C blue <newline>"
    using the next\2, linestart\2, lineend\2, tokenlength\2 predicates
    """
    token = c_type("token")
    num = c_type("num")

    next = c_pred("next",2,("token","token"))
    linestart = c_pred("linestart",2,("token","token"))
    lineend = c_pred("lineend",2,("token","token"))
    tokenlength = c_pred("tokenlength",2,("token","num"))

    n1 = c_const("n1",num)
    n3 = c_const("n3",num)
    n4 = c_const("n4",num)
    n5 = c_const("n5",num)
    node1 = c_const("node1",token)
    node2 = c_const("node2",token)
    node3 = c_const("node3",token)
    red = c_const("red",token)
    green = c_const("green",token)
    blue = c_const("blue",token)
    a_c = c_const("a_c",token)
    b_c = c_const("b_c",token)
    c_c = c_const("c_c",token)
    start = c_const("c_START",token)
    end = c_const("c_END",token)
    
    bk = Knowledge(
        next(start,node1),next(node1,a_c),next(a_c,red),
        next(red,node2),next(node2,green),next(green,b_c),
        next(b_c,node3),next(node3,c_c),next(c_c,blue),
        next(blue,end),tokenlength(node1,n4),tokenlength(node2,n4),
        tokenlength(node3,n4),tokenlength(a_c,n1),tokenlength(b_c,n1),
        tokenlength(c_c,n1),tokenlength(red,n3),tokenlength(green,n5),
        tokenlength(blue,n4),linestart(node1,node1),linestart(a_c,node1),
        linestart(red,node1),linestart(node2,node2),linestart(b_c,node2),
        linestart(green,node2),linestart(node3,node3),linestart(c_c,node3),
        linestart(blue,node3),lineend(node1,a_c),lineend(a_c,red),
        lineend(node2,red),lineend(b_c,green),lineend(node3,blue),
        lineend(c_c,blue),lineend(red,red),lineend(green,green),
        lineend(blue,blue))

    solver = SWIProlog()
    eval_fn1 = Coverage(return_upperbound=True)
    learner = Aleph(solver,eval_fn1,max_body_literals=3,do_print=False) 

    # 1. Consider the hypothesis: f1(word) :- word is the second word on a line
    if True:
        f1 = c_pred("f1",1,[token])
        neg = {f1(x) for x in [node1,node2,node3,blue,green,red]}
        pos = {f1(x) for x in [a_c,b_c,c_c]}
        task = Task(positive_examples=pos, negative_examples=neg)
        
        res = learner.learn(task,bk,None)
        print(res)

    # 2. Consider the hypothesis: f2(word) :- word is the first word on a line
    if True:
        f2 = c_pred("f2",1,[token])
        neg = {f1(x) for x in [a_c,b_c,c_c,blue,green,red]}
        pos = {f1(x) for x in [node1,node2,node3]}
        task2 = Task(positive_examples=pos, negative_examples=neg)

        res = learner.learn(task2,bk,None)
        print(res)

    # 3. Assume we have learned the predicate node(X) before (A, B and C and nodes).
    # We want to learn f3(Node,X) :- X is the next token after Node
    if True:
        node = c_pred("node",1,[token])
        color = c_pred("color",1,[token])
        nodecolor = c_pred("nodecolor",2,[token,token])
        a = c_var("A",token)
        b = c_var("B",token)
        bk_old = bk.get_all()
        bk = Knowledge(*bk_old, node(a_c),node(b_c),node(c_c),
                    node(a_c), node(b_c),node(c_c),
                    color(red),color(green),color(blue))
        pos = {nodecolor(a_c,red),nodecolor(b_c,green),nodecolor(c_c,blue)}
        neg = set()
        neg = {nodecolor(node1,red),nodecolor(node2,red),nodecolor(node3,red),
               nodecolor(node1,blue),nodecolor(node2,blue),nodecolor(node2,blue),
               nodecolor(node1,green),nodecolor(node2,green),nodecolor(node3,green),
               nodecolor(a_c,green),nodecolor(a_c,blue),nodecolor(b_c,blue),
               nodecolor(b_c,red),nodecolor(c_c,red),nodecolor(c_c,green)
            }
        task3 = Task(positive_examples=pos, negative_examples=neg)

        # prog = learner.learn(task3,bk,None,initial_clause=Body(node(a),color(b)))
        result = learner.learn(task3,bk,None,initial_clause=Body(node(a),color(b)),minimum_freq=3)
        print(result)
            
        

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

    learner = Aleph(solver,eval_fn,max_body_literals=4,do_print=False)
    learner2 = Aleph(solver,eval_fn2,max_body_literals=4,do_print=False)
    learner3 = Aleph(solver,eval_fn3,max_body_literals=4,do_print=False)

    result = learner.learn(task,background,None)
    print(result)

    # pr = learner2.learn(task,background,None)
    # print("Final program: {}".format(str(pr)))

    # pr = learner3.learn(task,background,None)
    # print("Final program: {}".format(str(pr)))


if __name__ == "__main__":
    learn_simpsons()
    learn_with_constants()
    learn_text()





    
    



