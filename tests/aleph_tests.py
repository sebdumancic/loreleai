from loreleai.learning.learners import Aleph
from loreleai.learning.task import Task,Knowledge
from loreleai.language.commons import c_type,Clause, Term, Variable, Constant, c_pred,c_const,c_var
from loreleai.learning import HypothesisSpace
from loreleai.reasoning.lp.prolog import SWIProlog
from loreleai.learning.eval_functions import Coverage, Compression

if __name__ == "__main__":
    

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

    pr = learner.learn(task,background,None)
    print("Final program: {}".format(str(pr)))

    pr = learner2.learn(task,background,None)
    print("Final program: {}".format(str(pr)))

    pr = learner3.learn(task,background,None)
    print("Final program: {}".format(str(pr)))




    
    



