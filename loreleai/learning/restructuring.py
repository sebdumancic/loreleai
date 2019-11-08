from functools import reduce
from itertools import combinations
from typing import Set, Dict, List, Tuple

from loreleai.language.commons import _are_two_set_of_literals_identical
from loreleai.language.lp import Theory, Clause, Predicate, Atom, Term


class Restructor:
    """
    Implements the theory restructuring functionality

    Args:
        max_literals (int): maximal number of literals to use in the restructuring clauses
        min_literals (int, optional): minimal number of literals to use in the restructuring clauses
    """

    def __init__(self, max_literals: int, min_literals: int = 2):
        self.max_literals = max_literals
        self.min_literals = min_literals
        self.candidate_counter = 0

    def _get_candidate_index(self):
        """
        Generates a unique index for a new latent head

        Returns:
            integer
        """
        self.candidate_counter += 1
        return self.candidate_counter

    def _get_candidates(self, clauses: Theory) -> Dict[Predicate, Set[Clause]]:
        """
        Extracts candidates for restructuring from the clauses in the theory

        Args:
            clauses: theory, a set of clauses to restructure

        Returns:
            a set of candidates represented as a dictionary:
                -- key: predicate
                -- value: all clauses having that predicate in the body
        """
        enumerated_bodies = {}

        def __create_latent_clause(literals: List[Atom]) -> Clause:
            head_name = f'latent{self._get_candidate_index()}'
            available_vars = {}
            for lit in literals:
                for v in lit.get_variables():
                    if v not in available_vars:
                        available_vars[v] = len(available_vars)
            available_vars = sorted(available_vars, key=lambda x: available_vars[x])
            head_pred = Predicate(head_name, len(available_vars), [x.get_type() for x in available_vars])

            return Clause(Atom(head_pred, available_vars), literals)

        def __process_candidates(accumulator: Dict[Predicate, Set[Clause]], clause: Clause) -> Dict[Predicate, Set[Clause]]:
            for length in range(self.min_literals, self.max_literals + 1):
                for cmb in combinations(clause.get_atoms(), length):
                    predicate_sig = set([x.get_predicate() for x in cmb])
                    predicate_sig = tuple(sorted(predicate_sig, key=lambda x: x.get_name()))

                    if predicate_sig not in enumerated_bodies:
                        # if no same predicate signature is available, just add the clause to all
                        enumerated_bodies[predicate_sig] = set()
                        enumerated_bodies[predicate_sig].add(tuple(cmb))

                        clause = __create_latent_clause(list(cmb))
                        for p in predicate_sig:
                            if p not in accumulator:
                                accumulator[p] = set()
                            accumulator[p].add(clause)
                    else:
                        # check if the same body has been already generated (in enumerated_bodies under the pred signature)
                        # if not, add the clause
                        if not any([_are_two_set_of_literals_identical(list(cmb), x) for x in enumerated_bodies[predicate_sig]]):
                            enumerated_bodies[predicate_sig].add(tuple(cmb))
                            clause = __create_latent_clause(list(cmb))
                            for p in predicate_sig:
                                if p not in accumulator:
                                    accumulator[p] = set()
                                accumulator[p].add(clause)

            return accumulator

        return reduce(__process_candidates, clauses.get_formulas(), {})

    def _encode_clause(self, clause: Clause, candidates: Dict[Predicate, Set[Clause]]):
        """
        Finds all possible encodings of the given clause by means of candidates

        Args:
            clause (Clause): a clause to encode
            candidates (Dict[Predicate, Set[Clause]): candidates to use to encode the provided clause
        """

        def __encode(atoms_to_cover: Set[Atom],
                     atoms_covered: Set[Atom],
                     atom_covering: Dict[Atom, Dict[Clause, Set[Tuple[List[Atom], Dict[Term, Term]]]]],
                     prefix=" "):

            if len(atoms_to_cover) == 0:
                return set()

            focus_atom = list(atoms_to_cover)[0]
            #print(f'{prefix}| focusing on {focus_atom}')

            matching_clauses = atom_covering[focus_atom].keys()
            #print(f'{prefix}|  found matching clauses {matching_clauses}')
            encodings = set()

            for cl in matching_clauses:
                for match in atom_covering[focus_atom][cl]:
                    #print(f'{prefix}|    processing clause {cl} with match {match}')
                    atms, sbs = match
                    new_atoms_to_cover = atoms_to_cover - set(atms) - {focus_atom}  # a problem somewhere here
                    new_atoms_covered = atoms_covered.union(atms)
                    #print(f'{prefix}|      atoms covered: {new_atoms_covered}; atoms to cover: {new_atoms_to_cover}')
                    encoding_rest = __encode(new_atoms_to_cover, new_atoms_covered, atom_covering, prefix=prefix*10)
                    #print(f'{prefix}|      encodings of the rest: {encoding_rest}')

                    if len(encoding_rest) == 0 and len(new_atoms_to_cover) == 0:
                        encodings.add(frozenset({cl.get_head().substitute(sbs)}))
                    else:
                        for enc_rest in encoding_rest:
                            encodings.add(enc_rest.union([cl.get_head().substitute(sbs)]))

            return encodings

        clause_predicates = clause.get_predicates()
        atoms_not_covered = clause.get_atoms()
        filtered_candidates = dict([(k, v) for (k, v) in candidates.items() if k in clause_predicates])

        # create index structure so that it is easy to get to the candidates that cover different atoms
        atom_to_covering_clause_index = {}
        for p in filtered_candidates:
            for cand in filtered_candidates[p]:
                for answer in cand.is_part_of(clause):
                    atms, sbs = answer
                    if len(sbs) == 0:
                        continue
                    for atm in atms:
                        if atm not in atom_to_covering_clause_index:
                            atom_to_covering_clause_index[atm] = {}
                        if cand not in atom_to_covering_clause_index[atm]:
                            atom_to_covering_clause_index[atm][cand] = []
                        if answer not in atom_to_covering_clause_index[atm][cand]:
                            atom_to_covering_clause_index[atm][cand].append(answer)

        return __encode(set(clause.get_atoms()), set(), atom_to_covering_clause_index)

    def _encode_theory(self, theory: Theory, candidates: Dict[Predicate, Set[Clause]]):
        """
        Encodes the entire theory with the provided candidates

        Args:
            theory (Theory): a set of clauses to encode
            candidates (Dict[Predicate, Set[Clause]]): clauses to use for encoding the theory

        """
        return dict([(x, self._encode_clause(x, candidates)) for x in theory.get_formulas()])

    def _find_redundancies(self, encoded_clauses: Dict[Clause, Set[Set[Atom]]]):
        """
        Identifies all redundancies in possible encodings

        Args:
            encoded_clauses (Dict[Clause, Set[Set[Atom]]]): encoded clauses
        """
        redundancy_counts = {}
        cooccurrence_counts = {}

        for cl in encoded_clauses:
            inner_counts = {}
            for enc in encoded_clauses[cl]:
                for l in range(2, len(enc) + 1):
                    for env_cmb in combinations(enc, l):
                        env_cmb = sorted(env_cmb, key=lambda x: x.get_predicate().get_name())

                        # order variables
                        var_indices = {}
                        for atm in env_cmb:
                            for v in atm.get_variables():
                                if v not in var_indices:
                                    var_indices[v] = len(var_indices)

                        # count coocurrences of latent predicates
                        pred_tuple = tuple([x.get_predicate().get_name() for x in env_cmb])
                        if pred_tuple not in cooccurrence_counts:
                            cooccurrence_counts[pred_tuple] = 0
                        cooccurrence_counts[pred_tuple] += 1

                        # create informative key (not depending on variable names)
                        env_cmb = tuple([f'{x.get_predicate().get_name()}({",".join([str(var_indices[y]) for y in x.get_variables()])})' for x in env_cmb])

                        if env_cmb not in inner_counts:
                            inner_counts[env_cmb] = 0
                        inner_counts[env_cmb] += 1

            for t in inner_counts:
                if t not in redundancy_counts:
                    redundancy_counts[t] = 0
                redundancy_counts[t] += 1

        return [k for k, v in redundancy_counts.items() if v > 1], [k for k, v in cooccurrence_counts.items() if v > 1]

    def restructure(self, clauses: Theory):
        """
        Starts the restructuring process

        Args:
            clauses: a theory to restructure

        Return:
            a new restructured theory
        """
        assert(all([isinstance(x, Clause) for x in clauses.get_formulas()]),
               "Restructuring only works with clausal theories")

        self.candidate_counter = 0

        all_candidates = self._get_candidates(clauses)

        clauses_encoded = self._encode_theory(clauses, all_candidates)

        redundancies, cooccurrences = self._find_redundancies(clauses_encoded)

        return redundancies