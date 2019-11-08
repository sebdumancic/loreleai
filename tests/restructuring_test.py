from loreleai.language.lp import parse, Theory
from loreleai.learning.restructuring import Restructor


class RestructuringMethods:

    def candidate_generation(self):
        clauses = ["t1(X,Y) :- a(X,Y), b(X,Y), c(X)",
                   "t2(X,Y) :- a(X,Y), b(Y,Z), c(Z)",
                   "t3(X,Y) :- b(X,Z), d(X, Y), c(X)"]

        clauses = [parse(x) for x in clauses]
        theory = Theory(clauses)

        restruct = Restructor(max_literals=2)

        cands = restruct._get_candidates(theory)

        assert len(cands) == 4

        all_cands = set()

        for p in cands:
            print(p, cands[p])
            all_cands = all_cands.union(cands[p])

        assert len(all_cands) == 8

    def simple_encoding_restructuring_space(self):
        clauses = ["t1(X,Y) :- a(X,Y), b(X,Y), c(X)"]
        clauses = [parse(x) for x in clauses]
        theory = Theory(clauses)

        restruct = Restructor(max_literals=2)
        all_cands = restruct._get_candidates(theory)
        cands = restruct._encode_theory(theory, all_cands)

        assert len(cands) > 0

        assert str(list(cands.keys())[0]) == str(clauses[0])

        assert len(cands[clauses[0]]) == 3

    def restructuring_encoding_theory(self):
        clauses = ["t1(X,Y) :- a(X,Y), b(X,Y), c(X)",
                   "t2(X,Y) :- a(X,Y), b(Y,Z), c(Z)",
                   "t3(X,Y) :- b(X,Z), d(X, Y), c(X)"]
        clauses = [parse(x) for x in clauses]
        theory = Theory(clauses)

        restruct = Restructor(max_literals=2)
        all_cands = restruct._get_candidates(theory)
        cands = restruct._encode_theory(theory, all_cands)

        assert len(cands) > 0

        assert len(cands) == 3

        distinct_candidates = set()
        for p in all_cands:
            distinct_candidates = distinct_candidates.union(all_cands[p])

        assert len(distinct_candidates) == 8

        assert all([len(cands[x]) == 3 for x in cands])

    def restructuring_no_redundancy(self):
        clauses = ["t1(X,Y) :- a(X,Y), b(X,Y), c(X)",
                   "t2(X,Y) :- a(X,Y), b(Y,Z), c(Z)",
                   "t3(X,Y) :- b(X,Z), d(X,Y), c(X)"]

        clauses = [parse(x) for x in clauses]
        theory = Theory(clauses)

        restruct = Restructor(max_literals=2)
        all_cands = restruct._get_candidates(theory)
        cands = restruct._encode_theory(theory, all_cands)

        redunds, _ = restruct._find_redundancies(cands)

        assert len(redunds) == 0

    def restructuring_redundancy(self):
        clauses = ["t1(X,Y) :- a(X,Y), b(X,Y), c(X)",
                   "t2(X,Y) :- a(X,Y), b(Y,Z), c(Z)",
                   "t3(X,Y) :- a(X,Y), b(X,Y), d(X,Z), c(X)"]

        clauses = [parse(x) for x in clauses]
        theory = Theory(clauses)

        restruct = Restructor(max_literals=2)

        all_cands = restruct._get_candidates(theory)
        cands = restruct._encode_theory(theory, all_cands)

        redunds, _ = restruct._find_redundancies(cands)

        assert len(redunds) == 3


def test_restructuring():
    test = RestructuringMethods()

    test.candidate_generation()
    test.simple_encoding_restructuring_space()
    test.restructuring_encoding_theory()
    test.restructuring_no_redundancy()
    test.restructuring_redundancy()

