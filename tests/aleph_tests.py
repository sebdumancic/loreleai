from loreleai.learning.learners import Aleph
from loreleai.language.commons import c_type,Clause, Term, Variable, Constant, c_pred,c_const,c_var

if __name__ == "__main__":
    
    learner = Aleph()

    x = c_var("X")
    y = c_var("Y")
    z = c_var("Z")

    # ancestor = c_pred("ancestor",2)
    # parent = c_pred("parent",2)

    # rose = c_const("rose")
    # luc = c_const("luc")
    # leo = c_const("leo")

    polygon = c_pred("polygon",1)
    rectangle = c_pred("rectangle",1)
    square = c_pred("square",1)
    pos = c_pred("pos",1)
    red = c_pred("red",1)

    theory = [
        Clause(polygon(x),[rectangle(x)]),
        Clause(rectangle(x),[square(x)]),
    ]
    c = Clause(pos(x),[red(x),square(x)])

    # print(type(Clause(parent(leo,rose),[]).get_head().get_arguments()[0]))

    print(learner._compute_bottom_clause(theory,c))



    
    



