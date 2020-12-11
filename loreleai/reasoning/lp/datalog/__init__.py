# from loreleai.reasoning.lp.datalog.muz import MuZ
# from .datalogsolver import DatalogSolver

from pylo.engines.datalog import MuZ
from pylo.engines.datalog.datalogsolver import DatalogSolver

__all__ = [
    'MuZ',
    'DatalogSolver'
]