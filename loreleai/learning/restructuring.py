import logging
import sys
from functools import reduce
from itertools import combinations
from typing import Set, Dict, List, Tuple, Iterator, Union, Sequence

from ortools.sat.python import cp_model

from loreleai.language.commons import _are_two_set_of_literals_identical, global_context, _create_term_signatures
from loreleai.language.lp import Clause, Predicate, Atom, Term, ClausalTheory, are_variables_connected, Variable

NUM_PREDICATES = 1
NUM_LITERALS = 2


class Restructor:
    """
    Implements the theory restructuring functionality

    Args:
        max_literals (int): maximal number of literals to use in the restructuring clauses
        min_literals (int, optional): minimal number of literals to use in the restructuring clauses
        head_variable_selection (int, optional): how to select variables for the head of the latent predicate
                                                 1 - take all
                                                 2 - take max_arity vars
        max_arity (int, optional): the number of variables to take in the head of the latent predicates when
                                    head_variable_selection = 2
    """

    def __init__(self, max_literals: int, min_literals: int = 2, head_variable_selection: int = 2, max_arity: int = 2,
                 minimise_redundancy=False, exact_redundancy=False, exclude_alternatives=False,
                 objective_type=NUM_PREDICATES, exclude_redundant_cands=False, reject_singleton=True,
                 logl=logging.INFO, logfile: str = None, logger=None):
        self.max_literals = max_literals
        self.min_literals = min_literals
        self._objective_type = objective_type
        self.aux_candidate_counter = 0
        self.head_variable_selection_strategy = head_variable_selection
        self.max_arity = max_arity
        self.enumerated_bodies = {}
        self.enumerated_body_signatures = {}
        self.candidate_usage_count = {}
        self.minimise_redundancy = minimise_redundancy
        self.minimise_redundancy_absolute_count = exact_redundancy
        self.reject_singletons = reject_singleton
        self._candidate_exclusion = []
        self.exclude_alternatives = exclude_alternatives
        self.exclude_redundant_candidates = exclude_redundant_cands
        self.redundant_candidates = []
        self.count_candidates = 0
        self.equals_zero = None
        self.log_level = logl

        # logging setup
        self._logger = logger if logger else logging.getLogger(logfile if logfile else '')

        if logfile is not None:
            log_file = logging.FileHandler(logfile)
            log_file.setLevel(logl)
            log_file.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
            self._logger.addHandler(log_file)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logl)
            console_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
            self._logger.addHandler(console_handler)

        self._logger.setLevel(logl)

    def _get_candidate_index(self):
        """
        Generates a unique index for a new latent head

        Returns:
            integer
        """
        self.aux_candidate_counter += 1
        return self.aux_candidate_counter

    def __create_latent_clause(self, literals: List[Atom], variable_strategy: int = 1, max_arity: int = 2) -> List[Clause]:
        if not are_variables_connected(literals):
            # if the variables are not connected in a graph, that makes it an invalid candidate
            return []

        head_name = f'latent{self._get_candidate_index()}'
        available_vars = {}
        for lit in literals:
            for v in lit.get_variables():
                if v not in available_vars:
                    available_vars[v] = len(available_vars)
        available_vars = sorted(available_vars, key=lambda x: available_vars[x])

        if variable_strategy == 1 or len(available_vars) == max_arity:
            # take all variables or number of variables is equal to max arity
            head_pred = global_context.predicate(head_name, len(available_vars), [x.get_type() for x in available_vars])
            # Predicate(head_name, len(available_vars), [x.get_type() for x in available_vars])
            atom = global_context.atom(head_pred, available_vars)  # Atom(head_pred, available_vars)
            cl = Clause(atom, literals)

            if self.reject_singletons and cl.has_singleton_var():
                return []
            else:
                self.count_candidates += 1
                return [cl]
        elif variable_strategy == 2:
            # need to select a subset of variables
            clauses = []

            for ind, var_cmb in enumerate(combinations(available_vars, max_arity)):
                # head_pred = Predicate(f'{head_name}_{ind + 1}', len(var_cmb), [x.get_type() for x in var_cmb])
                head_pred = global_context.predicate(f'{head_name}_{ind + 1}', len(var_cmb), [x.get_type() for x in var_cmb])
                atom = global_context.atom(head_pred, list(var_cmb))  # Atom(head_pred, list(var_cmb))
                cl = Clause(atom, literals)

                if self.reject_singletons and cl.has_singleton_var():
                    pass
                else:
                    self.count_candidates += 1
                    clauses.append(cl)

            # remember the alternatives, and add constraint that only one of these can be taken
            # assumes that all candidates have unique heads
            if self.exclude_alternatives and len(clauses) > 1:
                self._candidate_exclusion.append([x.get_head().get_predicate().get_name() for x in clauses])

            return clauses
        else:
            raise Exception(f'Unknown head variable selection strategy {variable_strategy}')

    def __process_candidates(self, accumulator: Dict[Predicate, Set[Clause]], clause: Clause) -> Dict[Predicate, Set[Clause]]:
        for length in range(self.min_literals, self.max_literals + 1):
            for cmb in combinations(clause.get_atoms(), length):
                cmb = list(cmb)
                # predicate_sig =  set([x.get_predicate() for x in cmb])
                predicate_sig = tuple(sorted([x.get_predicate() for x in cmb], key=lambda x: x.get_name()))

                potential_matches = [x for x in self.enumerated_bodies.get(predicate_sig, []) if len(x) == len(cmb)]

                if not any([_are_two_set_of_literals_identical(cmb, self.enumerated_body_signatures[x]) for x in potential_matches]):

                    if predicate_sig not in self.enumerated_bodies:
                        self.enumerated_bodies[predicate_sig] = set()

                    self.enumerated_bodies[predicate_sig].add(tuple(cmb))
                    self.enumerated_body_signatures[tuple(cmb)] = _create_term_signatures(cmb)

                    clauses = self.__create_latent_clause(list(cmb), self.head_variable_selection_strategy, self.max_arity)
                    for cl in clauses:
                        for p in predicate_sig:
                            if p not in accumulator:
                                accumulator[p] = set()
                            accumulator[p].add(cl)

        return accumulator

    def _get_candidates(self, clauses: Union[ClausalTheory, Sequence[Clause]]) -> Dict[Predicate, Set[Clause]]:
        """
        Extracts candidates for restructuring from the clauses in the theory

        Args:
            clauses: theory, a set of clauses to restructure

        Returns:
            a set of candidates represented as a dictionary:
                -- key: predicate
                -- value: all clauses having that predicate in the body
        """
        self._logger.info("Enumerating candidates...")

        return reduce(self.__process_candidates,
                      clauses.get_formulas() if isinstance(clauses, ClausalTheory) else clauses,
                      {})

    def __encode(self, atoms_to_cover: Sequence[Atom],
                 atoms_covered: Set[Atom],
                 atom_covering: Dict[Atom, Dict[Clause, Set[Tuple[List[Atom], Dict[Term, Term]]]]],
                 target_clause_head_vars: Set[Variable],
                 prefix=" ") -> Tuple[Set[Set[Atom]], Set[Clause]]:
        """
        Encoding of a set of atoms
        :param atoms_to_cover:
        :param atoms_covered:
        :param atom_covering:
        :param prefix:
        :return:
        """

        if len(atoms_to_cover) == 0:
            return set(), set()

        focus_atom = atoms_to_cover[0]
        # self._logger.debug(f'{prefix}| focusing on {focus_atom}')

        matching_clauses = atom_covering[focus_atom].keys()
        used_clauses = set()
        # print(f'{prefix}|  found matching clauses {matching_clauses}')
        encodings = set()

        for cl in matching_clauses:
            for match in atom_covering[focus_atom][cl]:
                # self._logger..debug(f'{prefix}|    processing clause {cl} with match {match}')
                atms, sbs = match  # subs: key - variables in cl, value -- variables to use as the substitutions (from )
                new_atoms_to_cover = [x for x in atoms_to_cover if x not in atms and x != focus_atom]
                new_atoms_covered = atoms_covered.union(atms)

                # make sure that none of the variables that would be kicked out are needed in the rest of the body
                retained_variables = set([sbs[x] for x in cl.get_head().get_variables()])
                kicked_out_variables = reduce((lambda x, y: x + y), [x.get_variables() for x in atms], [])
                kicked_out_variables = [x for x in kicked_out_variables if x not in retained_variables]

                if len(new_atoms_to_cover):
                    variables_in_the_rest_of_the_body = reduce((lambda x, y: x.union(y)), [x.get_variables() for x in new_atoms_to_cover], set()).union(target_clause_head_vars)
                else:
                    variables_in_the_rest_of_the_body = target_clause_head_vars

                if any([x in variables_in_the_rest_of_the_body for x in kicked_out_variables]):
                    continue
                else:
                    used_clauses.add(cl)
                    # self._logger.debug(f'{prefix}|      atoms covered: {new_atoms_covered}; atoms to cover: {new_atoms_to_cover}')
                    encoding_rest, inner_used = self.__encode(new_atoms_to_cover, new_atoms_covered, atom_covering, target_clause_head_vars.union(retained_variables), prefix=prefix * 2)
                    used_clauses = used_clauses.union(inner_used)
                    # self._logger.debug(f'{prefix}|      encodings of the rest: {encoding_rest}')

                    if len(encoding_rest) == 0 and len(new_atoms_to_cover) == 0:
                        encodings.add(frozenset({cl.get_head().substitute(sbs)}))
                    else:
                        for enc_rest in encoding_rest:
                            encodings.add(enc_rest.union([cl.get_head().substitute(sbs)]))

        return encodings, used_clauses

    def _encode_clause(self, clause: Clause,
                       candidates: Dict[Predicate, Set[Clause]],
                       originating_clause: Clause) -> List[Clause]:
        """
        Finds all possible encodings of the given clause by means of candidates

        Args:
            clause (Clause): a clause to encode
            candidates (Dict[Predicate, Set[Clause]): candidates to use to encode the provided clause
        """
        self._logger.warning(f'\tencoding clause {clause}')

        clause_predicates = clause.get_predicates()
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

        encoding, used_clauses = self.__encode(clause.get_atoms(), set(), atom_to_covering_clause_index, set(clause.get_head().get_variables()))
        encoding = [Clause(clause.get_head(), list(x)) for x in encoding if len(x) < len(clause)]  # if refactoring does not reduce the length on th clause, reject it
        for cl in encoding:
            cl.add_property("parent", originating_clause if originating_clause else clause)

        # update candidate counts
        for cl in used_clauses:
            if cl not in self.candidate_usage_count:
                self.candidate_usage_count[cl] = 0
            self.candidate_usage_count[cl] += 1

        return list(encoding)

    def _prune_candidate_set(self, candidates: Dict[Predicate, Set[Clause]]) -> Tuple[Dict[Predicate, Set[Clause]], Set[Clause]]:
        """
        Prunes the set of candidates; removes all candidates for which
            length(candidate) * usage(candidate) < length(candidate) + usage(candidate)

        Returns:
            in set: candidates to keep
            out set: pruned candidates
        """

        return dict([(k, set([x for x in v if len(x) * self.candidate_usage_count.get(x, 0) > len(x) + self.candidate_usage_count.get(x, 0)])) for k, v in candidates.items()]), \
                reduce((lambda x, y: x.union(y)), [set([x for x in v if len(x) * self.candidate_usage_count.get(x, 0) <= len(x) + self.candidate_usage_count.get(x, 0)]) for k, v in candidates.items()], set())

    def _encode_theory(self, theory: Union[ClausalTheory, Sequence[Clause]],
                       candidates: Dict[Predicate, Set[Clause]],
                       originating_clause: Clause = None) -> Dict[Clause, Sequence[Clause]]:
        """
        Encodes the entire theory with the provided candidates

        Args:
            theory (Theory): a set of clauses to encode
            candidates (Dict[Predicate, Set[Clause]]): clauses to use for encoding the theory

        """
        self._logger.info(f'Encoding theory...')
        return dict([(x, self._encode_clause(x, candidates, originating_clause if originating_clause else x)) for x in (theory.get_formulas() if isinstance(theory, ClausalTheory) else theory)])

    def _find_redundancies(self, encoded_clauses: Dict[Clause, Sequence[Clause]]) -> Tuple[Dict[Sequence[str], Sequence[Clause]], Sequence[Sequence[str]]]:
        """
        Identifies all redundancies in possible encodings

        Args:
            encoded_clauses (Dict[Clause, Set[Set[Atom]]]): encoded clauses
        """
        self._logger.info(f'Finding redundancies...')

        redundancy_counts = {}
        cooccurrence_counts = {}

        for cl in encoded_clauses:
            inner_counts = {}

            for enc_cl in encoded_clauses[cl]:
                enc = enc_cl.get_atoms()
                for l in range(2, len(enc) + 1):
                    for env_cmb in combinations(enc, l):
                        if not are_variables_connected(env_cmb):
                            continue

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
                            inner_counts[env_cmb] = []
                        inner_counts[env_cmb].append(enc_cl)

            for t in inner_counts:
                if t not in redundancy_counts:
                    redundancy_counts[t] = []
                redundancy_counts[t] += inner_counts[t]

        return dict([(k, v) for k, v in redundancy_counts.items() if len(v) > 1]), [k for k, v in cooccurrence_counts.items() if v > 1]

    def _find_candidate_redundancies(self, candidates: Dict[Predicate, Set[Clause]]):
        """
            Finds all redundancies in refactoring candidates and adds them to the self._candidate_exclusion

        """
        if self.min_literals == self.max_literals:
            pass
        else:
            all_predicates = candidates.keys()
            for length in range(2 if self.min_literals == 2 else self.min_literals, self.max_literals):
                for cmb in combinations(all_predicates, length):
                    cands = reduce((lambda x, y: x.union(y)), [candidates[p] for p in cmb])
                    cands_exact_length = [x for x in cands if len(x) == length]
                    cands_more_length = [x for x in cands if len(x) > length]

                    redundancies = [[x.get_head().get_predicate()] + [y.get_head().get_predicate() for y in cands_more_length if x.is_part_of(y)] for x in cands_exact_length]
                    self.redundant_candidates = [tuple([p.get_name() for p in x]) for x in redundancies]

    def __create_var_map(self, model: cp_model.CpModel,
                         candidates: Set[Clause],
                         co_occurrences: Sequence[Sequence[str]],
                         clause_dependencies: Dict[str, Sequence[str]]):
        """
        Creates a CP-SAT variable for (1) each candidate clause and (2) an auxiliary variable for each combination of
        candidate clauses that appear in the encodings of clauses

        Also create the equivalences between aux variables and the original ones

        Args:
            model (CpModel): an instance of CP-SAT model
            candidates (Set[Clause]): a set of clauses defining latent predicates
            co_occurrences (List[Iterator[str]]): List of latent predicate that co occur in many encodings
                        each co-occurrence should be represented as a tuple of strings (names of predicates)

        Returns:
            Dict[Union[predicate_name, Tuple[predicate_names]], cp-sat variable]
        """
        variable_map = {}

        aux_var_index = 1

        for cand in candidates:
            # TODO: use entire clause as the variable key
            variable_map[cand.get_head().get_predicate().get_name()] = model.NewBoolVar(cand.get_head().get_predicate().get_name())

        for co in co_occurrences:
            variable_map[co] = model.NewBoolVar(f'aux{aux_var_index}')
            # create the equality relating the aux variable to a product of
            model.AddMultiplicationEquality(variable_map[co], [variable_map[x] for x in co])

            # increase the index for the next aux var created
            aux_var_index += 1

        # add clause dependencies over different levels
        for cl in clause_dependencies:
            # add new var that will be true if all predicates from the body are selected
            b = model.NewBoolVar(f'aux_cldep_{self._get_candidate_index()}')
            model.AddBoolAnd([variable_map[c] for c in clause_dependencies[cl]]).OnlyEnforceIf(b)
            # selection of level+1 predicate implies selecting the predicates it depends on
            model.AddImplication(variable_map[cl], b)

        return variable_map

    def __impose_encoding_constraints(self, model: cp_model.CpModel,
                                      encodings: Dict[Clause, Sequence[ClausalTheory]],
                                      variable_map: Dict[Union[str, Iterator[str]], cp_model.IntVar]) -> Tuple[Dict[Clause, Sequence[cp_model.IntVar]], Dict[Clause, cp_model.IntVar]]:

        clause_level_selection_vars = {}
        encoding_clauses_to_vars = {}

        for clind, cl in enumerate(encodings):
            encs = encodings[cl]
            encs = [x.get_formulas() for x in encs]   # each item is the encoding level, as list of formulas

            clause_level_selection_vars[cl] = [model.NewBoolVar(f'aux_level_{x+1}_{self._get_candidate_index()}') for x in range(len(encs) + 1)] # + 1 to get 'choose not refactoring'
            level_components = []

            # individual_encodings = []

            for l_ind, level in enumerate(encs):
                individual_encodings_at_current_level = []

                for eind, en in enumerate(level):
                    plain_vars = sorted([a.get_predicate().get_name() for a in en.get_atoms()])

                    # find all sub-components that can be substituted with an aux variable from co-occurences
                    combs = []
                    for l in range(2, len(plain_vars)):
                        combs += list(combinations(plain_vars, l))

                    combs = [tuple(x) for x in combs if tuple(x) in variable_map]

                    # remove variables that are handled through the auxiliary variables
                    all_in_aux = set()
                    for aux in combs:
                        all_in_aux = all_in_aux.union(aux)

                    plain_vars = [x for x in plain_vars if x not in all_in_aux]

                    # add product to individual encodings
                    plain_vars = [variable_map[x] for x in plain_vars]
                    combs = [variable_map[x] for x in combs]

                    # ___ new encoding
                    tmp_var = model.NewBoolVar(f'ind_enc_{clind}_{l_ind+1}_{eind}')

                    # encoding with And/Or
                    # encodes one possible refactoring and adds them all together in individual_encodings_at_current_level
                    model.AddBoolAnd(plain_vars + combs).OnlyEnforceIf(tmp_var)
                    model.AddBoolOr([x.Not() for x in (plain_vars + combs)]).OnlyEnforceIf(tmp_var.Not())

                    individual_encodings_at_current_level.append(tmp_var)

                    # add the var to the corresponding clause
                    encoding_clauses_to_vars[encs[l_ind][eind]] = tmp_var

                # ENCODING WITH THE CURRENT LEVEL ONLY
                # takes individual encodings at the current level and makes an OR of all of them
                encoding_exists_var = model.NewBoolVar(f'aux_enc_exists_{self._get_candidate_index()}')
                model.AddBoolOr(individual_encodings_at_current_level).OnlyEnforceIf(encoding_exists_var)

                # And(level, Or(and individual encodings)) enforce only in entire_level_component (indicates that something from that level should be selected)
                level = clause_level_selection_vars[cl][l_ind+1] # l_ind = 0 is not refactoring
                entire_level_component = model.NewBoolVar(f'aux_levcom_{self._get_candidate_index()}')
                model.AddBoolAnd([level, encoding_exists_var]).OnlyEnforceIf(entire_level_component)
                level_components.append(entire_level_component)

            # at least one encoding has to be selected (or no encoding at all)
            model.AddBoolOr(level_components + [clause_level_selection_vars[cl][0]])

            # exactly one encoding level has to be selected
            model.Add(sum(clause_level_selection_vars[cl]) == 1)

        return clause_level_selection_vars, encoding_clauses_to_vars

    def __eliminate_redundancy_in_solutions(self, model: cp_model.CpModel,
                                            redundancies: Dict[int, Dict[Sequence[str], Sequence[Clause]]],
                                            variable_map: Dict[Union[str, Iterator[str]], cp_model.IntVar],
                                            encoded_clause_vars: Dict[Clause, cp_model.IntVar],
                                            encoding_level_vars: Dict[Clause, Sequence[cp_model.IntVar]]):

        for level in redundancies:
            for redundancy_pattern in redundancies[level]:
                all_with_pattern = []
                for cl in redundancies[level][redundancy_pattern]:
                    # + 1 because level=0 is no refactoring
                    corresponding_level_var = encoding_level_vars[cl.get_property("parent")][level + 1]
                    b = model.NewBoolVar(f'aux_red_{self._get_candidate_index()}')
                    model.AddBoolAnd([corresponding_level_var, encoded_clause_vars[cl]]).OnlyEnforceIf(b)
                    model.AddBoolOr([corresponding_level_var.Not(), encoded_clause_vars[cl].Not()]).OnlyEnforceIf(b.Not())
                    all_with_pattern.append(b)
                model.Add(sum(all_with_pattern) <= 1)

    def __eliminate_candidate_alternatives(self, model: cp_model.CpModel,
                                           variable_map: Dict[Union[str, Iterator[str]], cp_model.IntVar]):
        """
        Eliminates candidate alternatives:
            when multiple clause have the same body, but different head variables
            imposes the constaint that at most one of those can be selected


        Args:
            model (cp_model.CpModel): model
            variable_map (Dict[Union[str, Iterator[str]], cp_model.IntVar]): mapping from clauses to cp_model.vars
        """
        for alt in self._candidate_exclusion:
            model.Add(sum([variable_map[x] for x in alt]) <= 1)

    def __eliminate_redundant_candidates(self, model: cp_model.CpModel,
                                         variable_map: Dict[Union[str, Iterator[str]], cp_model.IntVar]):
        """
        Eliminates redundant candidates in the solution (imposes the same constraint __eliminate_candidate_alternatives)

        """
        for alt in self.redundant_candidates:
            # model.AddBoolOr([variable_map[x].Not() for x in alt])
            model.Add(sum([variable_map[x] for x in alt]) <= 1)

    def __set_objective(self, model: cp_model.CpModel,
                        candidates: Set[Clause],
                        variable_map: Dict[Union[str, Iterator[str]], cp_model.IntVar],
                        redundancies: Dict[int, Dict[Sequence[str], Sequence[Clause]]],
                        encoded_clause_vars: Dict[Clause, cp_model.IntVar],
                        encoding_level_vars: Dict[Clause, Sequence[cp_model.IntVar]],
                        encodings: Dict[Clause, Sequence[ClausalTheory]]):
        vars_to_use = [x.get_head().get_predicate().get_name() for x in candidates]
        vars_to_use = [variable_map[x] for x in vars_to_use]

        if self._objective_type == NUM_PREDICATES:
            if self.minimise_redundancy:
                # creating redundnancy indicator variables

                if self.minimise_redundancy_absolute_count:
                    # if we are minimising the exact count, we need the zero var
                    self.equals_zero = model.NewBoolVar('equals_zero')
                    model.Add(self.equals_zero == 0)

                individual_redunds = []
                for encoding_level in redundancies:
                    for redundancy_pattern in redundancies[encoding_level]:
                        all_with_pattern = []
                        for cl in redundancies[encoding_level][redundancy_pattern]:
                            corresponding_level_var = encoding_level_vars[cl.get_property("parent")][encoding_level+1] # +1 to account for 0 being no refactoring
                            b = model.NewBoolVar(f'aux_red_{self._get_candidate_index()}')
                            model.AddBoolAnd([corresponding_level_var, encoded_clause_vars[cl]]).OnlyEnforceIf(b)
                            model.AddBoolOr([corresponding_level_var.Not(), encoded_clause_vars[cl].Not()]).OnlyEnforceIf(b.Not())
                            all_with_pattern.append(b)

                        if self.minimise_redundancy_absolute_count:
                            out_b = model.NewIntVar(-1, len(all_with_pattern), f'aux_red_sum_{self._get_candidate_index()}')
                            model.Add((sum(all_with_pattern) - 1) == out_b)  # -1 allows to use 1 of the potentially redundant clauses, if only 1 is used there is no redundancy
                            b_max = model.NewBoolVar(f'aux_redmax_{self._get_candidate_index()}')
                            model.AddMaxEquality(b_max, [self.equals_zero, out_b])
                            individual_redunds.append(b_max)
                        else:
                            out_b = model.NewBoolVar(f'aux_red_sum_{self._get_candidate_index()}')
                            model.Add(sum(all_with_pattern) <= 1).OnlyEnforceIf(out_b.Not())  # reasoning inverted because out_b=0 means no redundancy
                            model.Add(sum(all_with_pattern) > 1).OnlyEnforceIf(out_b)
                            individual_redunds.append(out_b)

                model.Minimize(sum(vars_to_use + individual_redunds))
            else:
                model.Minimize(reduce((lambda x, y: x + y), vars_to_use))
        elif self._objective_type == NUM_LITERALS:
            all_weighted_clauses = []
            # lengths of selected clauses
            for cl in encodings:
                for ind, eth in enumerate(encodings[cl]):
                    wcl = [(x, len(x) + 1) for x in eth.get_formulas()]  # + 1 to include the head predicate
                    # sum_at_current_level = model.NewIntVar(0, sum([v for k, v in wcl]), f'aux_sum_{self._get_candidate_index()}')
                    # model.Add(reduce((lambda x, y: x + y), [encoded_clause_vars[k]*v for k, v in wcl]) == sum_at_current_level)
                    # sub_component = model.NewIntVar(0, sum([v for k, v in wcl]),  f'aux_comp_{self._get_candidate_index()}')
                    # model.AddProdEquality(sub_component, [sum_at_current_level, encoding_level_vars[cl][ind+1]]) # +1 to account for 0 being no refactoring
                    # all_weighted_clauses.append(sub_component)

                    level = encoding_level_vars[cl][ind+1]
                    for f, cost in wcl:
                        b = model.NewBoolVar(f'aux_and_{self._get_candidate_index()}')
                        model.AddBoolAnd([level, encoded_clause_vars[f]]).OnlyEnforceIf(b)
                        model.AddBoolOr([level.Not(), encoded_clause_vars[f].Not()]).OnlyEnforceIf(b.Not())
                        all_weighted_clauses.append(b*cost)

                # add no refactoring cost, encoding/refactoring level = 0
                all_weighted_clauses.append(encoding_level_vars[cl][0]*(len(cl) + 1))

            # lengths of selected candidates
            # exclude 1-length candidates because they help with partial refactoring:
            #                   the heads can simply be replaced by the body
            candidate_lengths = [(x.get_head().get_predicate().get_name(), len(x) + 1) for x in candidates if len(x) > 1]
            candidate_lengths = [variable_map[k]*v for k, v in candidate_lengths]

            model.Minimize(reduce((lambda x, y: x + y), all_weighted_clauses + candidate_lengths))

        else:
            raise Exception(f'unknown objective function {self._objective_type}')

    def _map_to_solver_and_solve(self, candidates: Set[Clause],
                                 encodings: Dict[Clause, Sequence[ClausalTheory]],
                                 redundancies: Dict[int, Dict[Sequence[str], Sequence[Clause]]],
                                 cooccurrences: Sequence[Sequence[str]],
                                 clause_dependencies: Dict[str, Sequence[str]],
                                 max_predicates,
                                 num_threads,
                                 max_time_s):
        """
        Maps the refactoring problem to CP-SAT and solves it

        Args:
            candidates (Set[Clause]): refactoring candidates
            encodings (Dict[Clause, Sequence[ClausalTheory]]): possible encodings of each a clause, per level
            redundancies (Dict[int, Dict[Sequence[str], Sequence[Clause]]]): redundandies per level
                                Dict[key: encoding level
                                     value: Dict[key: redundancy pattern (sequence of predicate names)
                                                 val: Sequence[Clauses] with the redundancy]

            cooccurrences (Sequence[Sequence[str]]):
                            tuples of predicate variables that often co-occur and can be replaced with a single variable
            clause_dependencies (Dict[str, Sequence[str]]): dependence of predicates over different encoding levels

        """

        self._logger.info(f'Mapping to CP and solving')

        model = cp_model.CpModel()
        variable_map = self.__create_var_map(model, candidates, cooccurrences, clause_dependencies)
        cls_level_indicators, encoded_cls_var = self.__impose_encoding_constraints(model, encodings, variable_map)

        if self.exclude_alternatives:
            self.__eliminate_candidate_alternatives(model, variable_map)

        if self.exclude_redundant_candidates:
            self.__eliminate_redundant_candidates(model, variable_map)

        if not self.minimise_redundancy and self._objective_type == NUM_PREDICATES:
            # if we want to eliminate redundancy and we are minimizing the number of predicates
            self.__eliminate_redundancy_in_solutions(model, redundancies, variable_map, encoded_cls_var, cls_level_indicators)
        self.__set_objective(model, candidates, variable_map, redundancies if self.minimise_redundancy else (), encoded_cls_var, cls_level_indicators, encodings)

        if max_predicates:
            model.Add(reduce((lambda x, y: x + y), [variable_map[x.get_head().get_predicate().get_name()] for x in candidates]) <= max_predicates)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = num_threads
        if max_time_s:
            solver.parameters.max_time_in_seconds = max_time_s

        solution_callback = VarArraySolutionPrinter([variable_map[x.get_head().get_predicate().get_name()] for x in candidates], self._logger)
        self._logger.info("Started solving")
        status = solver.SolveWithSolutionCallback(model, solution_callback)  # solver.Solve(model)
        self._logger.info(f"Solving done; status: {status}")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN):
            selected_clauses = set([k for k, v in variable_map.items() if isinstance(k, str) and solver.Value(v) == 1])
            selected_clauses = [x for x in candidates if x.get_head().get_predicate().get_name() in selected_clauses]
            refactoring_steps_per_clause = {}
            for cl in cls_level_indicators:
                tmp = [solver.Value(x) for x in cls_level_indicators[cl]]
                refactoring_steps_per_clause[cl] = tmp.index(1)  # + 1 - not needed because index  would mean 1 step of refactoring
            return selected_clauses, refactoring_steps_per_clause
        else:
            raise Exception('Could not find a satisfiable solution!')

    def _prepare_final_theory(self, clauses: ClausalTheory,
                              refactoring_predicates: Dict[Predicate, Set[Clause]],
                              refactoring_steps: Dict[Clause, int]) -> ClausalTheory:
        """
        Produces the final refactored theory

        Args:
            clauses (ClausalTheory): theory to refactor
            refactoring_predicates (Dict[Predicate, Set[Clause]]): encoding predicates to use
            refactoring_steps (Dict[Clause, int]]): number of refatoring steps for each clause in the theory

        Returns:
            refactorized theory

        """
        self._logger.setLevel(logging.CRITICAL)
        final_theory = list(reduce((lambda x, y: x.union(y)), [[p for p in x if len(p) > 1] for x in refactoring_predicates.values()], set()))

        # resolve single literal clauses
        single_body_cands = list(reduce((lambda x, y: x.union(y)), [[p for p in c if len(p) == 1] for c in refactoring_predicates.values()], set()))
        single_mapping = dict([(x.get_head().get_predicate(), list(x.get_predicates())[0]) for x in single_body_cands])
        while any([x in single_mapping for x in single_mapping.values()]):
            for item in single_mapping:
                if single_mapping[item] in single_mapping:
                    single_mapping[item] = single_mapping[single_mapping[item]]

        for cl in clauses.get_formulas():
            steps = refactoring_steps[cl]
            tmp_frm = cl

            while steps > 0:
                if not isinstance(tmp_frm, list):
                    tmp_frm = [tmp_frm]
                re_frm = []
                if len(tmp_frm) > 1:
                    for itm in tmp_frm:
                        try:
                            re_frm += self._encode_theory([itm], refactoring_predicates).values()
                        except Exception:
                            pass
                else:
                    re_frm = self._encode_theory(tmp_frm, refactoring_predicates)
                    re_frm = [x for x in re_frm.values()]
                re_frm = reduce((lambda x, y: x + y), re_frm)

                tmp_frm = [x for x in re_frm]
                steps -= 1

            if not isinstance(tmp_frm, list):
                tmp_frm = [tmp_frm]

            final_theory += tmp_frm

        # if any atoms refer to single literal clause
        new_frms = []
        for frm_itm in final_theory:
            if any([x in single_mapping for x in frm_itm.get_predicates()]):
                tmp_head = frm_itm.get_head()
                tmp_body = []
                for atm in frm_itm.get_atoms():
                    if atm.get_predicate() in single_mapping:
                        tmp_body.append(global_context.atom(single_mapping[atm.get_predicate()], atm.get_terms()))
                    else:
                        tmp_body.append(atm)

                new_frms.append(Clause(tmp_head, tmp_body))
            else:
                new_frms.append(frm_itm)

        self._logger.setLevel(self.log_level)

        return ClausalTheory(new_frms)

    def restructure(self, clauses: ClausalTheory, max_layers=None, max_predicate=None, num_threads=1, max_time_s=None,
                    prune_candidates=False):
        """
        Starts the restructuring process

        Args:
            clauses: a theory to restructure

        Return:
            a new restructured theory
        """
        self.aux_candidate_counter = 0
        self.count_candidates = 0
        candidatesPruned = set()

        # 4 -- optimal 2 -- feasible  0 -- unknown  3 -- infeasible
        # print(cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN, cp_model.INFEASIBLE)

        encodings_space: Dict[Clause, List[ClausalTheory]] = dict([(f, []) for f in clauses.get_formulas()])
        all_refactoring_candidates: Dict[Predicate, Set[Clause]] = {}
        all_redundancies = {}
        # ^-  Dict[ key: encoding level, to be used to select the appropriate encoding level var )
        #           val: Dict[ key: redundancy string (tuple)
        #                      val: encoded clauses Sequence[Clause]       ] ]
        all_cooccurences = []
        clause_dependencies = {}

        something_to_refactor: bool = True
        iteration_counter = 0

        while something_to_refactor:
            self.candidate_usage_count = {}
            self.enumerated_body_signatures = {}
            self._logger.info(f"\tStarting iteration: {iteration_counter}")
            # collect clauses to focus on
            if iteration_counter == 0:
                focus_clauses = list(encodings_space.keys())
            else:
                focus_clauses = reduce((lambda x, y: x + y),
                                       [encodings_space[x][iteration_counter-1].get_formulas() for x in encodings_space if len(encodings_space[x]) == iteration_counter], [])
            focus_clauses = [x for x in focus_clauses if len(x) >= self.min_literals]

            iteration_candidates = self._get_candidates(focus_clauses)

            #self._find_candidate_redundancies(iteration_candidates)
            # save the current candidates to the global collection
            # all_refactoring_candidates.update(iteration_candidates)

            if iteration_counter > 0:
                for p in iteration_candidates:
                    for cl in iteration_candidates[p]:
                        head_p = cl.get_head().get_predicate().get_name()
                        dep_ps = [x.get_predicate().get_name() for x in cl.get_atoms()]
                        if head_p not in clause_dependencies:
                            clause_dependencies[head_p] = dep_ps

            self._logger.info(f"\t\tfound {self.count_candidates} candidates")

            iteration_formulas = {}

            # extend the encoding of each original clause with the new layer of encodings
            if iteration_counter == 0:
                encoded_clauses = self._encode_theory(clauses, iteration_candidates)
                iteration_formulas.update(encoded_clauses)
                for cl in focus_clauses:
                    frms_to_add = [v for v in encoded_clauses[cl]]
                    if len(frms_to_add):
                        encodings_space[cl].append(ClausalTheory(frms_to_add))
            else:
                for cl in encodings_space:
                    if len(encodings_space[cl]) == iteration_counter:
                        encoded_clauses = self._encode_theory([x for x in encodings_space[cl][iteration_counter-1].get_formulas() if len(x) >= self.min_literals], iteration_candidates, originating_clause=cl)
                        if len(encoded_clauses) == 0:
                            continue
                        iteration_formulas.update(encoded_clauses)
                        # add encodings of all of the clauses
                        frms_to_add = reduce((lambda x, y: x + y), [v for k, v in encoded_clauses.items()], [])
                        if len(frms_to_add):
                            encodings_space[cl].append(ClausalTheory(frms_to_add))
                    else:
                        pass

            # if pruning is required
            # has to be done after the encoding as that is where the counts happen
            if prune_candidates:
                iteration_candidates, rejectedCandidates = self._prune_candidate_set(iteration_candidates)
                rejectedPredicates = set(sorted([x.get_head().get_predicate() for x in rejectedCandidates], key=lambda x: str(x)))
                false_exclusions = set()
                for cl in encodings_space:
                    if len(encodings_space[cl]) > iteration_counter:
                        formulas_to_remove = encodings_space[cl][-1].get_formulas(rejectedPredicates)

                        # if no refactoring is left after removing rejected predicates,
                        #       retain the rejected predicates that were used
                        if len(formulas_to_remove) == len(encodings_space[cl][-1]):
                            used_preds = reduce((lambda x, y: x.union(y)), [x.get_predicates() for x in formulas_to_remove], set())
                            false_exclusions = false_exclusions.union(rejectedPredicates.intersection(used_preds))

                # add false rejections to the iterations candidates
                matching_falsely_rejected_candidates = [x for x in rejectedCandidates if x.get_head().get_predicate() in false_exclusions]
                for item in matching_falsely_rejected_candidates:
                    for p in item.get_predicates():
                        iteration_candidates[p].add(item)

                rejectedPredicates = rejectedPredicates.difference(false_exclusions)
                candidatesPruned = candidatesPruned.union(rejectedCandidates.difference(matching_falsely_rejected_candidates))
                all_refactoring_candidates.update(iteration_candidates)

                # clean the refactored theories
                for cl in encodings_space:
                    if len(encodings_space[cl]) > iteration_counter:
                        encodings_space[cl][-1].remove_formulas_with_predicates(rejectedPredicates)

                # clear the clause dependencies
                for rej_can in rejectedPredicates:
                    if rej_can.get_name() in clause_dependencies:
                        del clause_dependencies[rej_can.get_name()]

                # clear alternatives
                rejectedPredicates = set([x.get_name() for x in rejectedPredicates])
                self._candidate_exclusion = [x for x in self._candidate_exclusion if not any([p in rejectedPredicates for p in x])]
            else:
                all_refactoring_candidates.update(iteration_candidates)

            # find candidate redundancies
            if self.exclude_redundant_candidates:
                self._logger.info("\t Finding redundancies amongst candidates")
                self._find_candidate_redundancies(iteration_candidates)
                self._logger.info("\t\t\t done!")

            if self._objective_type == NUM_PREDICATES:
                # detect all redundancies and co-occurences
                tmp_redundancies, tmp_cooccurrences = self._find_redundancies(iteration_formulas)
                all_cooccurences += tmp_cooccurrences
                all_redundancies[iteration_counter] = tmp_redundancies

            iteration_counter += 1

            if iteration_counter == max_layers or len(focus_clauses) == 0:
                something_to_refactor = False

        distinct_candidates = set()
        for p in all_refactoring_candidates:
            distinct_candidates = distinct_candidates.union(all_refactoring_candidates[p])

        self._logger.info(f"Found {self.count_candidates} candidates in total; pruned {len(candidatesPruned)}")

        selected_clauses, refactoring_steps = self._map_to_solver_and_solve(distinct_candidates, encodings_space,
                                                                            all_redundancies, all_cooccurences, clause_dependencies,
                                                                            max_predicate, num_threads, max_time_s)

        # create_clause_index
        cl_ind = {}
        for cl in selected_clauses:
            pps = cl.get_predicates()
            for p in pps:
                if p not in cl_ind:
                    cl_ind[p] = set()
                cl_ind[p].add(cl)

        final_theory = self._prepare_final_theory(clauses, cl_ind, refactoring_steps)

        return selected_clauses, final_theory


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, logger):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self._logger = logger

    def on_solution_callback(self):
        self.__solution_count += 1
        self._logger.info(f"\tIteration {self.__solution_count}: objective {self.ObjectiveValue()}, selected {sum([1 for x in self.__variables if self.Value(x) == 1])}")

    def solution_count(self):
        return self.__solution_count