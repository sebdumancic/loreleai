from loreleai.learning.learners import Aleph
from loreleai.language.commons import c_type,Clause, Term, Variable, Constant, c_pred,c_const,c_var

if __name__ == "__main__":
    
    learner = Aleph()

    x = c_var("X")
    y = c_var("Y")
    z = c_var("Z")

    ancestor = c_pred("ancestor",2)
    parent = c_pred("parent",2)

    rose = c_const("rose")
    luc = c_const("luc")
    leo = c_const("leo")

    theory = [
        Clause(ancestor(x,y),[parent(x,y)]),
        Clause(ancestor(x,y),[parent(x,z),ancestor(z,y)]),
        Clause(parent(rose,luc),[]),
        Clause(parent(leo,rose),[])
    ]

    # print(type(Clause(parent(leo,rose),[]).get_head().get_arguments()[0]))

    model = learner._herbrand_model(theory)
    print(model)



    
    



