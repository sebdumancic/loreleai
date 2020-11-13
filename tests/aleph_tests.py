from loreleai.learning.learners import Aleph
from loreleai.language.commons import c_type,Clause, Term, Variable, Constant, c_pred,c_const,c_var

if __name__ == "__main__":
    learner = Aleph()
    learner.learn(None,None,None)

    x = c_var("X")
    y = c_var("Y")

    polygon = c_pred('polygon',1)
    square = c_pred('square',1)
    red = c_pred('red',1)
    rectangle = c_pred('rectangle',1)
    pos = c_pred('pos',1)
    pos2 = c_pred('pos2',2)

    bk = [
        polygon(x) <= rectangle(x),
        rectangle(x) <= square(x)
    ]

    c = pos(x) <= red(x) & square(x)
    c2 = pos2(x,y) <= red(x) & square(y)

    c_sk,_ = learner._skolemize(c2)
    print(c_sk.negate())



