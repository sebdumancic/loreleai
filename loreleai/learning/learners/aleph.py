import typing
from abc import ABC, abstractmethod
from typing import Sequence, Dict
from orderedset import OrderedSet

from loreleai.learning.abstract_learners import Learner,TemplateLearner
from loreleai.learning import Task, Knowledge, HypothesisSpace, TopDownHypothesisSpace
from loreleai.language.commons import Clause, Constant,c_type,Variable,Not,Atom, Procedure
from itertools import product, combinations_with_replacement
from collections import Counter
from loreleai.reasoning.lp import LPSolver
from loreleai.learning.language_manipulation import plain_extension
from loreleai.learning.language_filtering import has_singleton_vars, has_duplicated_literal
from loreleai.learning.eval_functions import EvalFunction, Coverage
 

class Aleph(TemplateLearner):
    def __init__(self,solver: LPSolver, eval_fn: EvalFunction,max_body_literals=5,do_print=False):
        super().__init__(solver,eval_fn,do_print)
        self._max_body_literals = max_body_literals
        

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
        self._assert_knowledge(knowledge)

        examples_to_use = examples
        pos,_ = examples_to_use.get_examples()

        # List of clauses we're learning
        prog = []

        i=1

        while len(pos) > 0:
            # Pick example from pos
            pos_ex = Clause(list(pos)[0],[])
            bk = knowledge.as_clauses()
            print("Next iteration: generalizing example {}".format(str(pos_ex)))
            bottom = self._compute_bottom_clause(bk,pos_ex)
            
            # Predicates can only be picked from the body of the bottom clause
            body_predicates = list(set(map(lambda l: l.get_predicate(),bottom.get_body().get_literals())))
            hs = TopDownHypothesisSpace(primitives=[
                                            lambda x: plain_extension(x, pred, connected_clauses=True) for pred in body_predicates],
                                        head_constructor=pos_ex.get_head().get_predicate(),
                                        expansion_hooks_reject=[lambda x, y: has_singleton_vars(x, y),
                                                                lambda x, y: has_duplicated_literal(x, y)])
            
            cl = self._learn_one_clause(examples_to_use,hs)
            print("- New clause: "+str(cl))
            prog.append(cl)
            print(prog)

            # update covered positive examples
            covered = self._execute_program(cl)

            pos, neg = examples_to_use.get_examples()
            pos = pos.difference(covered)

            examples_to_use = Task(pos, neg)

            print("Finished iteration {}".format(i))
            print("Current program: {}".format(str(prog)))
            i+=1

        return prog

    def initialise_pool(self):
        self._candidate_pool = OrderedSet()

    def put_into_pool(self, candidates: typing.Union[Clause, Procedure, typing.Sequence]) -> None:
        if isinstance(candidates, Clause):
            self._candidate_pool.add(candidates)
        else:
            self._candidate_pool |= candidates

    def get_from_pool(self) -> Clause:
        return self._candidate_pool.pop(0)

    def stop_inner_search(self, eval: typing.Union[int, float], examples: Task, clause: Clause) -> bool:
        if eval > 0:
            return True
        else:
            return False

    def process_expansions(self, examples: Task, exps: typing.Sequence[Clause], hypothesis_space: TopDownHypothesisSpace) -> typing.Sequence[Clause]:
        # eliminate every clause with more body literals than allowed
        exps = [cl for cl in exps if len(cl) <= self._max_body_literals]

        # check if every clause has solutions
        exps = [(cl, self._solver.has_solution(*cl.get_body().get_literals())) for cl in exps]
        new_exps = []

        for ind in range(len(exps)):
            if exps[ind][1]:
                # keep it if it has solutions
                new_exps.append(exps[ind][0])
            else:
                # remove from hypothesis space if it does not
                hypothesis_space.remove(exps[ind][0])

        return new_exps

    def _execute_program(self, clause: Clause) -> typing.Sequence[Atom]:
        """
        Evaluates a clause using the Prolog engine and background knowledge

        Returns a set of atoms that the clause covers
        """
        if len(clause.get_body().get_literals()) == 0:
            return []
        else:
            head_predicate = clause.get_head().get_predicate()
            head_variables = clause.get_head_variables()

            sols = self._solver.query(*clause.get_body().get_literals())

            sols = [head_predicate(*[s[v] for v in head_variables]) for s in sols]

            return sols

    def _learn_one_clause(self, examples: Task, hypothesis_space: TopDownHypothesisSpace) -> Clause:
        """
        Learns a single clause

        Returns a clause
        """
        # reset the search space
        hypothesis_space.reset_pointer()

        # empty the pool just in case
        self.initialise_pool()

        # put initial candidates into the pool
        self.put_into_pool(hypothesis_space.get_current_candidate())
        current_cand = None
        score = -100

        while current_cand is None or (len(self._candidate_pool) > 0 and not self.stop_inner_search(score, examples, current_cand)):
            # get first candidate from the pool
            current_cand = self.get_from_pool()

            # expand the candidate
            _ = hypothesis_space.expand(current_cand)
            # this is important: .expand() method returns candidates only the first time it is called;
            #     if the same node is expanded the second time, it returns the empty list
            #     it is safer than to use the .get_successors_of method
            exps = hypothesis_space.get_successors_of(current_cand)

            exps = self.process_expansions(examples, exps, hypothesis_space)
            # add into pool
            self.put_into_pool(exps)

            score = self.evaluate(examples, current_cand)

        if self._print:
            print(f"- New clause: {current_cand}")
            print(f"- Candidates has value {round(score,2)} for metric '{self._eval_fn.name()}'")
        return current_cand


    def _compute_bottom_clause(self,theory: Sequence[Clause],c: Clause) -> Clause:
        """
        Computes the bottom clause given a theory and a clause.
        Algorithm from (De Raedt,2008)
        """
        
        # 1. Find a skolemization substitution θ for c (w.r.t. B and c)
        _, theta = self._skolemize(c)
        
        # 2. Compute the least Herbrand model M of theory ¬body(c)θ
        body_facts = [Clause(l.substitute(theta),[]) for l in c.get_body().get_literals()]
        m = self._herbrand_model(theory+body_facts)
        
        # 3. Deskolemize the clause head(cθ) <= M and return the result.
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
        # print(m[0])

        # Find a fact in the theory (i.e. no body literals)
        facts = list(filter(lambda c:len(c.get_body().get_literals())==0,clauses))
        if len(facts) == 0:
            raise AssertionError("Theory does not contain ground facts, which necessary to compute a minimal Herbrand model!")

        # print("Finished iteration 0")
        m[1] = list(facts)
        # print(m[1])

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
                        # print("Added fact {} to m[{}]".format(str(new_fact),i+1))
                        # print(m[i+1])

            
            # print(f"Finished iteration {i}")
            m[i+1] = list(set(m[i+1]+m[i]))
            # print("New model: "+str(m[i+1]))
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





        


    
    
