from loreleai.language.lp import c_var, c_pred, c_const, Predicate, Constant, Variable, Clause, Atom, Disjunction
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_singleton_vars, connected_clause
from loreleai.learning.language_manipulation import plain_extension, BottomClauseExpansion


class LanguageTest:

    def manual_constructs(self):
        p1 = c_const("p1")
        p2 = c_const("p2")
        p3 = c_const("p3")

        parent = c_pred("parent", 2)
        grandparent = c_pred("grandparent", 2)

        f1 = parent(p1, p2)
        f2 = parent(p2, p3)

        v1 = c_var("X")
        v2 = c_var("Y")
        v3 = c_var("Z")

        cl = grandparent(v1, v3) <= parent(v1, v2) & parent(v2, v3)

        assert isinstance(p1, Constant)
        assert isinstance(p2, Constant)
        assert isinstance(p3, Constant)

        assert isinstance(parent, Predicate)
        assert isinstance(grandparent, Predicate)

        assert isinstance(v1, Variable)
        assert isinstance(v2, Variable)
        assert isinstance(v2, Variable)

        assert isinstance(cl, Clause)

        assert isinstance(f1, Atom)
        assert isinstance(f2, Atom)

    def shorthand_constructs(self):
        parent = c_pred("parent", 2)
        grandparent = c_pred("grandparent", 2)

        f1 = parent("p1", "p2")
        f2 = parent("p2", "p3")
        f3 = grandparent("p1", "X")

        assert isinstance(parent, Predicate)
        assert isinstance(grandparent, Predicate)

        assert isinstance(f1, Atom)
        assert isinstance(f2, Atom)
        assert isinstance(f3, Atom)

        assert isinstance(f1.arguments[0], Constant)
        assert isinstance(f1.arguments[1], Constant)
        assert isinstance(f3.arguments[1], Variable)


class LanguageManipulationTest:

    def plain_clause_extensions(self):
        parent = c_pred("parent", 2)
        grandparent = c_pred("grandparent", 2)

        cl1 = grandparent("X", "Y") <= parent("X", "Z")

        extensions = plain_extension(cl1, parent, connected_clauses=False)

        assert len(extensions) == 16

    def plain_clause_extensions_connected(self):
        parent = c_pred("parent", 2)
        grandparent = c_pred("grandparent", 2)

        cl1 = grandparent("X", "Y") <= parent("X", "Z")

        extensions = plain_extension(cl1, parent, connected_clauses=True)

        assert len(extensions) == 15

    def plain_procedure_extension(self):
        parent = c_pred("parent", 2)
        ancestor = c_pred("ancestor", 2)

        cl1 = ancestor("X", "Y") <= parent("X", "Y")
        cl2 = ancestor("X", "Y") <= parent("X", "Z") & parent("Z", "Y")

        proc = Disjunction([cl1, cl2])

        extensions = plain_extension(proc, parent, connected_clauses=False)

        assert len(extensions) == 25


class HypothesisSpace():

    def top_down_plain(self):
        grandparent = c_pred("grandparent", 2)
        father = c_pred("father", 2)
        mother = c_pred("mother", 2)

        hs = TopDownHypothesisSpace(primitives=[lambda x: plain_extension(x, father),
                                                lambda x: plain_extension(x, mother)],
                                    head_constructor=grandparent)

        current_cand = hs.get_current_candidate()
        assert len(current_cand) == 3

        expansions = hs.expand(current_cand[0])
        assert len(expansions) == 6

        expansions_2 = hs.expand(expansions[0])
        assert len(expansions_2) == 10

        expansions3 = hs.expand(expansions[1])
        assert len(expansions3) == 32

        hs.block(expansions[2])
        expansions4 = hs.expand(expansions[2])
        assert len(expansions4) == 0

        hs.remove(expansions[3])
        expansions5 = hs.get_successors_of(current_cand[0])
        assert len(expansions5) == 5

        hs.move_pointer_to(expansions[1])
        current_cand = hs.get_current_candidate()
        assert current_cand[0] == expansions[1]

        hs.ignore(expansions[4])
        hs.move_pointer_to(expansions[4])
        expansions6 = hs.get_current_candidate()
        assert len(expansions6) == 0

    def top_down_limited(self):
        grandparent = c_pred("grandparent", 2)
        father = c_pred("father", 2)
        mother = c_pred("mother", 2)

        hs = TopDownHypothesisSpace(primitives=[lambda x: plain_extension(x, father, connected_clauses=False),
                                                lambda x: plain_extension(x, mother, connected_clauses=False)],
                                    head_constructor=grandparent,
                                    expansion_hooks_keep=[lambda x, y: connected_clause(x, y)],
                                    expansion_hooks_reject=[lambda x, y: has_singleton_vars(x, y)])

        current_cand = hs.get_current_candidate()
        assert len(current_cand) == 3

        expansions = hs.expand(current_cand[1])
        assert len(expansions) == 6

        expansion2 = hs.expand(expansions[1])
        assert len(expansion2) == 16

    def bottom_up(self):
        a = c_pred("a", 2)
        b = c_pred("b", 2)
        c = c_pred("c", 1)
        h = c_pred("h", 2)

        cl = h("X", "Y") <= a("X", "Z") & b("Z", "Y") & c("X")

        bc = BottomClauseExpansion(cl)

        hs = TopDownHypothesisSpace(primitives=[lambda x: bc.expand(x)],
                                    head_constructor=h)

        cls = hs.get_current_candidate()
        assert len(cls) == 3

        exps = hs.expand(cls[1])
        assert len(exps) == 2

        exps2 = hs.expand(exps[0])
        assert len(exps2) == 4


def test_language():
    test = LanguageTest()
    test.manual_constructs()

    test_bias = LanguageManipulationTest()
    test_bias.plain_clause_extensions()
    test_bias.plain_clause_extensions_connected()
    test_bias.plain_procedure_extension()

    test_hypothesis_space = HypothesisSpace()
    test_hypothesis_space.top_down_plain()
    test_hypothesis_space.top_down_limited()
    test_hypothesis_space.bottom_up()

test_language()