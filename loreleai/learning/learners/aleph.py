import typing
from abc import ABC, abstractmethod
from loreleai.learning.abstract_learners import Learner
from typing import Sequence, Dict

from loreleai.learning import Task, Knowledge, HypothesisSpace
from loreleai.language.commons import Clause, Constant,c_type,Variable,Not
from itertools import product, combinations_with_replacement
from collections import Counter

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

    def _compute_bottom_clause(self,theory: Sequence[Clause],c: Clause) -> Clause:
        """
        Algorithm from (De Raedt,2008)
        """
        
        #1. Find a skolemization substitution θ for c (w.r.t. B and c)
        _, theta = self._skolemize(c)
        
        #2. Compute the least Herbrand model M of theory ¬body(c)θ
        body_facts = [Clause(l.substitute(theta),[]) for l in c.get_body().get_literals()]
        m = self._herbrand_model(theory+body_facts)
        
        #3. Deskolemize the clause head(cθ) <= M and return the result.
        theta_inv = {value:key for key,value in theta.items()}
        return Clause(c.get_head(),[l.get_head().substitute(theta_inv) for l in m])

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

    def _herbrand_model(self,clauses: Sequence[Clause]) -> Sequence[Clause]:
        """
        Computes a minimal Herbrand model of a theory 'clauses'.
        Algorithm from (De Raedt, 2008)
        """
        i=1
        m = {0:[]}
        print(m[0])

        # Find a fact in the theory (i.e. no body literals)
        facts = list(filter(lambda c:len(c.get_body().get_literals())==0,clauses))
        if len(facts) == 0:
            raise AssertionError("Theory does not contain ground facts, which necessary to compute a minimal Herbrand model!")

        print("Finished iteration 0")
        m[1] = list(facts)
        print(m[1])

        while Counter(m[i]) != Counter(m[i-1]):
            model_constants = _flatten([fact.get_head().get_arguments() for fact in m[i]])

            m[i+1] = []
            rules = list(filter(lambda c:len(c.get_body().get_literals())>0,clauses))
            for rule in rules:
                # if there is a substition theta such that
                # all literals in rule._body are true in the previous model
                body = rule.get_body()
                body_vars = body.get_variables()

                # Build all substitutions body_vars -> model_constants
                substitutions = _all_maps(body_vars,model_constants)
                
                for theta in substitutions:
                    # add_rule is True unless there is some literal that never
                    # occurs in m[i]
                    add_fact = True
                    for body_lit in body.get_literals():
                        candidate = body_lit.substitute(theta)
                        facts = list(map(lambda x:x.get_head(),m[i]))
                        # print("Does {} occur in {}?".format(candidate,facts))
                        if candidate in facts:
                            pass
                            # print("Yes")
                        else:
                            add_fact = False

                    new_fact = Clause(rule.get_head().substitute(theta),[])
                    if add_fact and not new_fact in m[i+1] and not new_fact in m[i]:
                        m[i+1].append(new_fact)
                        print("Added fact {} to m[{}]".format(str(new_fact),i+1))
                        # print(m[i+1])

            
            print(f"Finished iteration {i}")
            m[i+1] = list(set(m[i+1]+m[i]))
            print("New model: "+str(m[i+1]))
            i += 1
        
        return m[i]

    
def _flatten(l) -> Sequence:
    """
    [[1],[2],[3]] -> [1,2,3]
    """
    return [item for sublist in l for item in sublist]

def _all_maps(l1,l2) -> Sequence[Dict[Variable,Constant]]:
    """
    Return all maps between l1 and l2
    such that all elements of l1 have an entry in the map
    """
    sols = []
    for c in combinations_with_replacement(l2,len(l1)):
        sols.append({l1[i]:c[i] for i in range(len(l1))})
    return sols





        


    
    
