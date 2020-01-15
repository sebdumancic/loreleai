import datetime
import math
import os
import random
import subprocess
import sys

import numpy as np
import scipy.stats as stats

num_trials = 10
num_tasks = 1000
playtimes = [200, 400, 600, 800, 1000] #, 400, 600, 800, 1000]  # list(range(0,2200,200))
trials = [1]  # list(range(1,num_trials+1))
systems = ['playgol','nopi']
max_right = 5
max_forwards = 5
ROOT_FILE = os.path.dirname(os.path.abspath(__file__)) + "/../.."
MIN_LITERALS = 2
MAX_LITERALS = 3
MAX_LAYERS = None
MAX_TIME_S = 3600
PRUNE = True
EXCLUDE_ALTERNATIVES = False
EXCLUDE_REDUNDANT_CANDS = True
EXCLUDE_REDUNDANCIES = True
MINIMISE_REDUNDANCIES = True
REJECT_SINGLETONS = True


experiment_files = \
"""
:-['../playgol'].
:-['lego-bk'].
:-['../metagol'].
"""

experiment_setup = \
"""

:- use_module(library(time)).

play_time_interval(2).
max_build_time(60).
max_play_depth(4).
max_build_depth(5).
cpus(4).
"""

experiment_primitives = \
"""
%% tell metagol to use the BK
prim(left/2).
prim(right/2).
prim(place1/2).
prim(at_start/1).
prim(not_at_start/1).
prim(at_end/1).
prim(not_at_end/1).
"""

experiment_rest = \
"""
%% metarules
metarule(ident,[P/2,Q/2],([P,A,B]:-[[Q,A,B]])).
metarule(precon,[P/2,Q/1,R/2],([P,A,B]:-[[Q,A],[R,A,B]])).
metarule(postcon,[P/2,Q/2,R/1],([P,A,B]:-[[Q,A,B],[R,B]])).
metarule(chain,[P/2,Q/2,R/2],([P,A,B]:-[[Q,A,C],[R,C,B]])).
metarule(tailrec,[P/2,Q/2],([P,A,B]:-[[Q,A,C],[P,C,B]])).

metagol:functional.

func_test([P,A,B],PS,Prog):-
  \+ (metagol:prove_deduce([[P,A,C]],PS,Prog),C\=B).

a:-
    cpus(CPU_COUNT),
    set_prolog_flag(cpu_count,CPU_COUNT),
    games(Games),
    playgol(Games),
    b,
    halt.

b:-
    cpus(CPU_COUNT),
    set_prolog_flag(cpu_count,CPU_COUNT),
    get_time(T),
    stamp_date_time(T, date(DY,DM,DD,TH,TM,TS,_,_,_), 'UTC'),
    format('% started solving build tasks at ~w ~w ~w ~w:~w:~w\\n', [DD, DM, DY, TH, TM, TS]),
    max_build_depth(BuildDepth),
    playgol:update_depth(BuildDepth),
    max_build_time(BuildTime),
    retractall(max_time(_)),
    assert(max_time(BuildTime)),
    tasks(Tasks),
    learn_tasks(Tasks,Progs),
    length(Progs,N),
    format('% num solved ~w\\n',[N]),
    get_time(T2),
    stamp_date_time(T2, date(DY2,DM2,DD2,TH2,TM2,TS2,_,_,_), 'UTC'),
    format('% finished solving build tasks at ~w ~w ~w ~w:~w:~w\\n', [DD2, DM2, DY2, TH2, TM2, TS2]),
    halt.

learn_tasks(Tasks,Progs):-
    concurrent_maplist(learn_aux,Tasks,Xs),
    findall(true,member(true,Xs),Progs).

learn_aux(T,true):-
    build_pos(T,Atom),
    Pos=[Atom],
    max_time(MaxTime),
    catch(call_with_time_limit(MaxTime,learn(Pos,[],Prog)),time_limit_exceeded,(writeln('%timeout'),false)),!,
    pprint(Prog).
learn_aux(_,false).

play_examples(T,Pos,[]):-
    findall(Atom,play_pos(T,Atom),Pos).

do_test:-
    tasks(Tasks),
    forall(member(Task,Tasks),(
        atomic_list_concat(['b',Task],Pred),
        (current_predicate(Pred/2) ->
            format('%solved,~w,~w\\n',[Task,1]);
            format('%solved,~w,~w\\n',[Task,0])))),
    halt.
"""


space_size = 6
play_size = [2, 4]
build_space = 6


def gen_grid(space_size):
    s = (space_size, space_size)
    a = np.random.randint(1, size=(1,space_size)).tolist()[0]
    b = np.random.randint(space_size+1, size=(1,space_size)).tolist()[0]
    return a, b


def gen_data_new():
    for k in trials:
        tasks = []
        for i in range(num_tasks):
            init, end = gen_grid(build_space)
            tasks.append(f'build_pos({i},b{i}(world(1,{build_space},{init}),world(_,{build_space},{end}))).\n')
        with open('data/build-{}.pl'.format(k),'w') as f:
            f.write('tasks({}).\n'.format(list(range(num_tasks))))
            for x in tasks:
                f.write(x)
        for playtime in playtimes:
            with open('data/play-{}-{}.pl'.format(playtime,k),'w') as f:
                f.write('games({}).\n'.format(list(range(playtime))))
                for i in range(playtime):
                    ps = random.randint(play_size[0], play_size[1])
                    init, end = gen_grid(ps)
                    f.write(f'play_pos({i},p{i}(world(1,{ps},{init}),world(_,{ps},{end}))).\n')


def gen_data():
    for k in trials:
        for n in num_tasks:
            tasks_ = [f't{i}' for i in range(n)]
            with open(f'data/tasks-{n}-{k}.pl', 'w') as f:
                f.write('\n'.join(tasks_))

            with open(f'data/train-{n}-{k}.pl', 'w') as f:
                for i in range(n):
                    (a,b) = gen_grid()
                    f.write(f'pos_ex(t{i},{a},{b}).\n')


def call_prolog(action,load_files,output):
    cmd = "load_files(['experiment',{}],[silent(true)]). ".format(','.join(load_files))
    cmd += '{}.'.format(action)
    with open(output, 'w') as outf:
        p = subprocess.Popen(['swipl','-q','-G8g','-T8g','-L8g'], stdin=subprocess.PIPE, stdout=outf)
        p.stdin.write(cmd.encode())
        print(cmd)
        (output, err) = p.communicate()


def play_and_buid():
    for k in trials:
        for p in playtimes:
            for s in systems:
                playf="'data/play-{}-{}'".format(p,k)
                buildf="'data/build-{}'".format(k)
                programf=f"programs/{s}/programs-{p}-{k}.pl"
                call_prolog('a',[playf,buildf],programf)


def prepare_play_as_background(refactored_play_file, output_file):
    f = open(refactored_play_file)
    outf = open(output_file, 'w')

    primitives = []
    for line in f.readlines():
        if len(line) > 3:
            head, body = line.strip().split(":-")
            outf.write(f"{head.replace('-', '_')} :- {body.replace('-', '_')}.\n")
            head = head.strip().replace(')', '(').split('(')
            head, args = head[0], head[1]
            primitives.append(f"{head}/{len(args.split(','))}")

    outf.close()
    f.close()

    return primitives


def generate_experiment_file(exp_file, refactored_primitives, primitives_to_add):
    f = open(exp_file, 'w')

    f.write(experiment_files + "\n")
    f.write(f":-['{refactored_primitives}'].\n")
    f.write("\n")

    f.write(experiment_setup + "\n\n")
    f.write(experiment_primitives + "\n")

    for prim in primitives_to_add:
        f.write(f"prim({prim}).\n")

    f.write(experiment_rest + "\n")

    f.close()


def call_prolog_new(action,load_files,output):
    cmd = "load_files([{}],[silent(true)]). ".format(','.join(load_files))
    cmd += '{}.'.format(action)
    with open(output, 'w') as outf:
        p = subprocess.Popen(['swipl','-q','-G8g','-T8g','-L8g'], stdin=subprocess.PIPE, stdout=outf)
        p.stdin.write(cmd.encode())
        print(cmd)
        (output, err) = p.communicate()


def build_p():
    print(f"running build_p; trials")
    # SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_singl{REJECT_SINGLETONS}'
    SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'
    if not os.path.exists("./programs_p"):
        os.mkdir("./programs_p")

    if not os.path.exists(f"./programs_p/{SETUP}"):
        os.mkdir(f"./programs_p/{SETUP}")

    for k in trials:
        for p in playtimes:
            print(f"trial {k} play {p}")
            playf = f"{ROOT_FILE}/refactored_programs_p/lego/{SETUP}/refactored_programs-{p}-{k}.pl" # "'data/play-{}-{}'".format(p,k)
            buildf = f"'{ROOT_FILE}/build/lego/build-{k}.pl'"   # "'data/build-{}'".format(k)
            programf = f"./programs_p/{SETUP}/programs-{p}-{k}.pl"  # f"programs/{s}/programs-{p}-{k}.pl"
            refprimf = f"./programs_p/{SETUP}/refactored_primitives-{p}-{k}.pl"  # f"programs/{s}/programs-{p}-{k}.pl"
            expf = f"./programs_p/{SETUP}/experiment-{p}-{k}.pl"  # f"programs/{s}/programs-{p}-{k}.pl"

            prims = prepare_play_as_background(playf, refprimf)
            generate_experiment_file(expf, refprimf, prims)

            files_to_load = [f"'{expf}'", f"'{refprimf}'", buildf]

            call_prolog_new('b', files_to_load, programf)


def build_pb():
    if not os.path.exists("./programs_pb"):
        os.mkdir("./programs_pb")

    for k in trials:
        for p in playtimes:
            playf = f"{ROOT_FILE}/refactored_programs_pb/lego/refactored_programs-{p}-{k}.pl" # "'data/play-{}-{}'".format(p,k)
            buildf = f"'{ROOT_FILE}/build/lego/build-{k}.pl'"   # "'data/build-{}'".format(k)
            programf = f"./programs_pb/programs-{p}-{k}.pl"  # f"programs/{s}/programs-{p}-{k}.pl"
            refprimf = f"./programs_pb/refactored_primitives-{p}-{k}.pl"  # f"programs/{s}/programs-{p}-{k}.pl"
            expf = f"./programs_pb/experiment-{p}-{k}.pl"  # f"programs/{s}/programs-{p}-{k}.pl"

            prims = prepare_play_as_background(playf, refprimf)
            generate_experiment_file(expf, refprimf, prims)

            files_to_load = [f"'{expf}'", f"'{refprimf}'", buildf]

            call_prolog_new('b', files_to_load, programf)


def test():
    for k in trials:
        for p in playtimes:
            for s in systems:
                buildf="'data/build-{}'".format(k)
                programf=f"'programs/{s}/programs-{p}-{k}.pl'"
                resultsf=f'results/{s}/{p}-{k}.pl'
                call_prolog('do_test',[buildf,programf],resultsf)


def test_new():
    SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'
    if not os.path.exists("./results_p"):
        os.mkdir("./results_p")

    if not os.path.exists(f"./results_p/{SETUP}"):
        os.mkdir(f"./results_p/{SETUP}")

    for k in trials:
        for p in playtimes:
            buildf = f"'{ROOT_FILE}/build/lego/build-{k}.pl'"
            programf = f"'./programs_p/{SETUP}/programs-{p}-{k}.pl'"
            resultsf = f'./results/{SETUP}/{p}-{k}.pl'
            expf = f"'./programs_p/{SETUP}/experiment-{p}-{k}.pl'"  # f"programs/{s}/programs-{p}-{k}.pl"
            #call_prolog('do_test',[buildf,programf],resultsf)

            call_prolog_new('do_test', [expf, buildf, programf], resultsf)


def get_accs(system,p):
    all_num_solved=[]
    all_accs=[]
    for k in trials:
        k_num_solved=[]
        fname = f'results/{system}/{p}-{k}.pl'
        with open(fname,'r') as f:
            for line in f:
                line=line.strip()
                xs=line.split(',')
                if len(xs) <2:
                    continue
                if line.startswith('%solved'):
                    k_num_solved+=[int(xs[2])]
                    all_accs.append(int(xs[2]))
                # else:
                #     all_accs+=[int(xs[1])]
        all_num_solved.append(np.mean(k_num_solved))
    return (np.mean(all_num_solved)*100,stats.sem(all_num_solved)*100,all_accs)


def mct(xs,ys):
    b = sum(1.0 for (x,y) in zip(xs,ys) if x == 1 and y == 0)
    c = sum(1.0 for (x,y) in zip(xs,ys) if x == 0 and y == 1)
    # print(b,c)
    McN = math.pow((b-c),2) / (b+c)
    print('P-value: %f'%(1-stats.chi2.cdf(McN,1)))


def results():
    system_accs = {}
    for system in systems:
        system_accs[system]=[]
        for p in playtimes:
            (num_solved,sem,all_accs) = get_accs(system,p)
            system_accs[system].extend(all_accs)
            print('({},{}) +- (0,{})'.format(p,round(num_solved,2),round(sem,2)))
    # xs=system_accs['playgol']
    # ys=system_accs['nopi']
    # mct(xs,ys)


def get_accs_new(setup, p):
    all_num_solved=[]
    all_accs=[]
    for k in trials:
        k_num_solved=[]
        fname = f'./results_p/{setup}/{p}-{k}.pl'
        with open(fname,'r') as f:
            for line in f:
                line=line.strip()
                xs=line.split(',')
                if len(xs) <2:
                    continue
                if line.startswith('%solved'):
                    k_num_solved+=[int(xs[2])]
                    all_accs.append(int(xs[2]))
                # else:
                #     all_accs+=[int(xs[1])]
        print({"system": "knorf", "playtask": p, "trial": k, "acuracy": np.mean(k_num_solved)})
        all_num_solved.append(np.mean(k_num_solved))
    return (np.mean(all_num_solved)*100,stats.sem(all_num_solved)*100,all_accs)


def results_new():
    SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'
    for p in playtimes:
        (num_solved, sem, all_accs) = get_accs_new(SETUP, p)
        #print('({},{}) +- (0,{})'.format(p, round(num_solved, 2), round(sem, 2)))


def parse_size(setup, p, k):
    playf = f"{ROOT_FILE}/refactored_programs_p/lego/{setup}/refactored_programs-{p}-{k}.pl.log"
    last_non_empty_line = ""
    fil = open(playf)
    for line in fil.readlines():
        if len(line) > 3:
            last_non_empty_line = line.strip()
    ori_size, refactor_size = int(last_non_empty_line.split()[6]), int(last_non_empty_line.split()[-1])
    fil.close()

    return ori_size, refactor_size


def theory_sizes():
    SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'
    sizes = []
    for p in playtimes:
        for t in trials:
            ori, ref = parse_size(SETUP, p, t)
            sizes.append({'playtasks': p, 'trial': t, 'original': ori, 'refactored': ref})
            print({'playtasks': p, 'trial': t, 'system': "knorf", 'theory_size': ref})
            print({'playtasks': p, 'trial': t, 'system': "metagol", 'theory_size': ori})

    return sizes


def parse_runtime(setup, p, k):
    programf = f"./programs_p/{setup}/programs-{p}-{k}.pl"
    start_time = None
    end_time = None
    fil = open(programf)

    for line in fil.readlines():
        if '% started solving build tasks' in line:
            tmp = line.strip().split()
            time = tmp[-1].split(":")
            start_time = datetime.datetime(day=int(tmp[6]), month=int(tmp[7]), year=int(tmp[8]), hour=int(time[0]),
                                           minute=int(time[1]), second=int(float(time[2])))
        if '% finished solving build tasks' in line:
            tmp = line.strip().split()
            time = tmp[-1].split(":")
            end_time = datetime.datetime(day=int(tmp[6]), month=int(tmp[7]), year=int(tmp[8]), hour=int(time[0]),
                                         minute=int(time[1]), second=int(float(time[2])))
    if start_time is None:
        start_time = datetime.datetime.fromtimestamp(os.path.getctime(programf))

    if end_time is None:
        end_time = datetime.datetime.fromtimestamp(os.path.getmtime(programf))

    return (end_time - start_time).total_seconds()


def runtimes():
    SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'
    runtimes = []
    for p in playtimes:
        for t in trials:
            seconds = parse_runtime(SETUP, p, t)
            runtimes.append({'type': 'refactored', 'playtasks': p, 'trial': t, 'runtime': seconds})
            print({"system": "knorf", "playtask": p, "trial": t, "runtime": seconds})

    return runtimes


def get_build_program_size(setup, p, k):
    programf = f"./programs_p/{setup}/programs-{p}-{k}.pl"
    count = 0
    fil = open(programf)

    for line in fil.readlines():
        if line.startswith('%') or len(line) < 3 or 'true' in line:
            continue
        elif line.startswith('b'):
            head, body = line.strip().split(':-')
            body = body.replace('.', ',')
            body = body.split('),')
            count += 1 + len(body)
        else:
            pass

    return count

def build_program_size():
    SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'
    runtimes = []
    for p in playtimes:
        for t in trials:
            seconds = get_build_program_size(SETUP, p, t)
            runtimes.append({'type': 'refactored', 'playtasks': p, 'trial': t, 'runtime': seconds})
            print({"system": "knorf", "playtask": p, "trial": t, "program_size": seconds})

    return runtimes

# play_and_buid()
# test()
# results()

cmd = sys.argv[1]
stp = sys.argv[2]

other_setup = stp  # expected to be in form parameter:value separated with white-space
other_setup = other_setup.strip().split()
other_setup = dict([tuple(c.split(":")) for c in other_setup])

if 'maxl' in other_setup:
    MAX_LITERALS = int(other_setup['maxl'])

if 'minl' in other_setup:
    MIN_LITERALS = int(other_setup['minl'])

if 't' in other_setup:
    if ',' not in other_setup['t']:
        trials = [int(other_setup['t'])]
    else:
        trials = [int(x) for x in other_setup['t'].split(',')]

if 'mlay' in other_setup:
    MAX_LAYERS = int(other_setup['mlay'])

if 'pt' in other_setup:
    if ',' not in other_setup['pt']:
        playtimes = [int(other_setup['pt'])]
    else:
        playtimes = [int(x) for x in other_setup['pt'].split(",")]

if 'mt' in other_setup:
    MAX_TIME_S = int(other_setup['mt'])

if 'p' in other_setup:
    PRUNE = bool(other_setup['p'])

if 'a' in other_setup:
    EXCLUDE_ALTERNATIVES = True if other_setup['a'] == 'true' else False

if 'rc' in other_setup:
    EXCLUDE_REDUNDANT_CANDS = bool(other_setup['rc'])

if 'rs' in other_setup:
    REJECT_SINGLETONS = bool(other_setup['rs'])

if 'rr' in other_setup:
    EXCLUDE_REDUNDANCIES = bool(other_setup['rr'])

if 'mr' in other_setup:
    MINIMISE_REDUNDANCIES = bool(other_setup['mr'])

if cmd == 'gen':
    gen_data_new()
if cmd == 'learn':
    play_and_buid()
if cmd == 'learn-build-p':
    build_p()
if cmd == 'learn-build-pb':
    build_pb()
if cmd == 'test':
    test()
if cmd == 'test-new':
    test_new()
if cmd == 'results':
    results()
if cmd == 'results-new':
    results_new()
if cmd == 'size':
    theory_sizes()
if cmd == 'runtime':
    runtimes()
if cmd == 'program-size':
    build_program_size()