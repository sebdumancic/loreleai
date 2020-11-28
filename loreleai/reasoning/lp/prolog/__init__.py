# from .GNUProlog import GNUProlog
# from .SWIProlog import SWIProlog
# from .XSBProlog import XSBProlog
# #from .Prolog import Prolog

engines = []

try:
    from pylo.engines.prolog import GNUProlog
    engines += ['GNUProlog']
except Exception:
    pass

try:
    from pylo.engines.prolog import SWIProlog
    engines += ['SWIProlog']
except Exception:
    pass

try:
    from pylo.engines.prolog import XSBProlog
    engines += ['XSBProlog']
except Exception:
    pass

engines += ['Prolog']
__all__ = engines

# __all__ = [
#     "SWIProlog",
#     "XSBProlog",
#     "GNUProlog",
#     "Prolog"
# ]