import datetime
import os
import sys

from loreleai.language.lp import ClausalTheory
from loreleai.learning.restructuring import Restructor, NUM_LITERALS

ROOT_FOLDER = os.path.abspath(os.path.dirname(__file__))
TRIAL = [1]
NUMBER_PLAY_TASKS = [200, 400, 600, 800, 1000]
THREADS = 4
MAX_TIME_S = 1 * 2 * 60    # hour * min * sec
MAX_LAYERS = None
MIN_LITERALS = 2
MAX_LITERALS = 2
PRUNE = True
EXCLUDE_ALTERNATIVES = False
EXCLUDE_REDUNDANT_CANDS = False
EXCLUDE_REDUNDANCIES = False
REJECT_SINGLETONS = True
MINIMISE_REDUNDANCIES = False


def refactor_theory(input_theory_file, output_theory_file, max_literals, min_literals, max_layers, max_time, prune, alternatives, redundant_cands, redundancies, reject_singletons, minimise_red):
    import logging
    logging.root.handlers = []

    theory = ClausalTheory(read_from_file=input_theory_file)
    unfolded_theory = theory.unfold()

    original_theory_preds = theory.get_predicates()
    unfolded_theory_preds = unfolded_theory.get_predicates()
    difference_preds = original_theory_preds.difference(unfolded_theory_preds)

    frms = unfolded_theory.get_formulas()

    unfolded_theory = ClausalTheory([x for x in frms if len(x) > 1])
    unfolded_theory.remove_duplicates()

    restructurer = Restructor(max_literals=max_literals, min_literals=min_literals,
                              max_arity=2, head_variable_selection=2,
                              logl=logging.INFO, logfile=f"{output_theory_file}.log",
                              minimise_redundancy=minimise_red, exact_redundancy=False, prevent_redundancies=redundancies,
                              exclude_alternatives=alternatives, objective_type=NUM_LITERALS,
                              exclude_redundant_cands=redundant_cands, reject_singleton=reject_singletons)

    restructurer._logger.info(f"Max number of predicates: {len(difference_preds)}")

    cls, thr = restructurer.restructure(unfolded_theory, max_layers=max_layers,
                                        max_predicate=len(difference_preds), num_threads=THREADS,
                                        max_time_s=max_time, prune_candidates=prune)

    out_th = open(output_theory_file, 'w')
    for frm in thr.get_formulas():
        out_th.write(str(frm) + "\n")

    out_th.close()

    theory.visualize(f'{output_theory_file}_visual_original.pdf', only_numbers=True)
    thr.visualize(f'{output_theory_file}_visual_refactored.pdf', only_numbers=True)

    restructurer._logger.info(f"Original theory has {len(difference_preds)} invented predicates, refactored one {len(cls)}")
    restructurer._logger.info(f"Original theory has {theory.num_literals()} literals, while refactored one {thr.num_literals()}")


DOMAIN = sys.argv[1]
setup = sys.argv[2]

other_setup = sys.argv[3]  # expected to be in form parameter:value separated with white-space
other_setup = other_setup.strip().split()
other_setup = dict([tuple(c.split(":")) for c in other_setup])

if 'maxl' in other_setup:
    MAX_LITERALS = int(other_setup['maxl'])

if 'minl' in other_setup:
    MIN_LITERALS = int(other_setup['minl'])

if 't' in other_setup:
    if ',' not in other_setup['t']:
        TRIAL = [int(other_setup['t'])]
    else:
        TRIAL = [int(x) for x in other_setup['t'].split(',')]

if 'mlay' in other_setup:
    MAX_LAYERS = int(other_setup['mlay'])

if 'pt' in other_setup:
    if ',' not in other_setup['pt']:
        NUMBER_PLAY_TASKS = [int(other_setup['pt'])]
    else:
        NUMBER_PLAY_TASKS = [int(x) for x in other_setup['pt'].split(",")]

if 'mt' in other_setup:
    MAX_TIME_S = int(other_setup['mt'])

if 'p' in other_setup:
    PRUNE = bool(other_setup['p'])

if 'a' in other_setup:
    EXCLUDE_ALTERNATIVES = True if other_setup['a'] == 'true' else False

if 'rc' in other_setup:
    EXCLUDE_REDUNDANT_CANDS = bool(other_setup['rc'])

if 'rr' in other_setup:
    EXCLUDE_REDUNDANCIES = bool(other_setup['rr'])

if 'mr' in other_setup:
    MINIMISE_REDUNDANCIES = bool(other_setup['mr'])

if 'thr' in other_setup:
    THREADS = int(other_setup['thr'])

if 'rs' in other_setup:
    REJECT_SINGLETONS = bool(other_setup['rs'])


PROGRAM_FOLDER = ROOT_FOLDER + f"/programs_{setup}"
OUTPUT_FOLDER = ROOT_FOLDER + f"/refactored_programs_{setup}"
SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

if not os.path.exists(f'{OUTPUT_FOLDER}/{DOMAIN}'):
    os.mkdir(f'{OUTPUT_FOLDER}/{DOMAIN}')

if not os.path.exists(f'{OUTPUT_FOLDER}/{DOMAIN}/{SETUP}'):
    os.mkdir(f'{OUTPUT_FOLDER}/{DOMAIN}/{SETUP}')

for num_plays in NUMBER_PLAY_TASKS:
    for trial in TRIAL:
        file_name = f'programs-{num_plays}-{trial}.pl'
        refactored_fn = f'refactored_{file_name}'

        print(f"[{datetime.datetime.now()}] working on {file_name}")
        print(f"                               refactoring to {refactored_fn}")

        refactor_theory(f'{PROGRAM_FOLDER}/{DOMAIN}/{file_name}', f'{OUTPUT_FOLDER}/{DOMAIN}/{SETUP}/{refactored_fn}',
                        MAX_LITERALS, MIN_LITERALS, MAX_LAYERS, MAX_TIME_S, PRUNE, EXCLUDE_ALTERNATIVES,
                        EXCLUDE_REDUNDANT_CANDS, EXCLUDE_REDUNDANCIES, REJECT_SINGLETONS, MINIMISE_REDUNDANCIES)
