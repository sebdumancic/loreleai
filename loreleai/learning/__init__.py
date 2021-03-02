from .hypothesis_space import HypothesisSpace, TopDownHypothesisSpace
from .language_filtering import has_singleton_vars, has_duplicated_literal, max_var, max_pred_occurrences, \
    connected_clause, connected_body, negation_at_the_end
from .language_manipulation import plain_extension
from .task import Knowledge, Interpretation, Task
from .utilities import FillerPredicate, are_variables_connected
from .abstract_learners import TemplateLearner
