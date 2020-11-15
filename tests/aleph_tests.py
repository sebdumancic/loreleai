from loreleai.learning.learners import Aleph
from loreleai.learning.task import Task,Knowledge
from loreleai.language.commons import c_type,Clause, Term, Variable, Constant, c_pred,c_const,c_var
from loreleai.learning import HypothesisSpace
from loreleai.reasoning.lp.prolog import SWIProlog
from loreleai.learning.eval_functions import Coverage

if __name__ == "__main__":
    

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
    neg = {grandparent("a", "b"), grandparent("a", "g"), grandparent("i", "j"),grandparent("f","g")}

    task = Task(positive_examples=pos, negative_examples=neg)

    # print(type(Clause(parent(leo,rose),[]).get_head().get_arguments()[0]))

    solver = SWIProlog()

    learner = Aleph(solver,Coverage(),max_body_literals=4,do_print=True)

    pr = learner.learn(task,background,None)
    # x = c_var("X")
    # print(learner.evaluate(task,Clause(grandparent(x,x),[])))
    print("Final program: {}".format(str(pr)))



    
    



