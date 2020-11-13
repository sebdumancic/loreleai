import typing
from abc import ABC, abstractmethod
from loreleai.learning.abstract_learners import Learner

from loreleai.learning import Task, Knowledge, HypothesisSpace
from loreleai.language.commons import Clause, Constant,c_type,Variable

class Aleph(Learner):
    def __init__(self):
        pass

    def learn(self, examples: Task, knowledge: Knowledge, hypothesis_space: HypothesisSpace):
        """
        To find a hypothesis, Aleph uses the following set covering approach:
        1.  Select a positive example to be generalised. If none exists, stop; otherwise proceed to the next step.
        2.  Construct the most specific clause (the bottom clause) (Muggleton, 1995) that entails theselected example and that is consistent with the mode declarations.
        3.  Search for a clause more general than the bottom clause and that has the best score.
        4.  Add the clause to the current hypothesis and remove all the examples made redundantby it.
        Return to step 1.
        (Description from Cropper and Dumancic )
        """
        print("Finished learning")

    def _skolemize(self,clause: Clause) -> Clause:
        # Find all variables in clause
        vars = clause.get_variables()

        # Map from X,Y,Z,... -> sk0,sk1,sk2,...
        subst = {vars[i]:Constant(f"sk{i}",c_type("thing")) for i in range(len(vars))}

        # Apply this substitution to create new clause without quantifiers
        b = []
        h = clause.get_head().substitute(subst)
        for lit in clause.get_body().get_literals():
            b.append(lit.substitute(subst))

        # Return both new clause and mapping
        return Clause(h,b),subst

        


    
    
