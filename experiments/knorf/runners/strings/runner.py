import datetime
import os
import random
import subprocess
import sys

import numpy as np
import scipy.stats as stats

num_trials=10
playtimes = [200, 400, 600, 800, 1000] #, 400, 600, 800, 1000]  # list(range(0,2200,200))
trials = [1]
systems=['playgol','nopi']
max_string_size = 20
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
:-['string-bk'].
:-['../metagol'].
"""

experiment_setup = \
"""

:- use_module(library(time)).

play_time_interval(2).
max_build_time(60).
max_play_depth(4).
max_build_depth(5).
cpus(2).

metagol:functional.
%% metagol:max_clauses(10).

func_test([P,s(In,Out1),s(_,[])],PS,G):-
  \+ (metagol :prove_deduce([[P,s(In,Out2),s(_,[])]],PS,G),Out1\= Out2).
"""

experiment_primitives = \
"""
%% tell metagol to use the BK
prim(is_empty/1).
prim(not_empty/1). % maybe?

prim(is_space/1).
prim(not_space/1). % maybe?

prim(is_uppercase/1).
prim(is_lowercase/1).

prim(is_letter/1).
prim(not_letter/1).

prim(is_number/1).
prim(not_number/1).

prim(copy1/2).
prim(skip1/2).
prim(mk_uppercase/2).
prim(mk_lowercase/2).
%% prim(write1/3).
"""


experiment_rest = \
"""
metarule(precon,[P/2,Q/1,R/2],([P,A,B]:-[[Q,A],[R,A,B]])).
metarule(postcon,[P/2,Q/2,R/1],([P,A,B]:-[[Q,A,B],[R,B]])).
metarule(chain,[P/2,Q/2,R/2],([P,A,B]:-[[Q,A,C],[R,C,B]])).
metarule(tailrec,[P/2,Q/2],([P,A,B]:-[[Q,A,C],[P,C,B]])).
%% metarule(curry3,[P/2,Q/3,C/0],([P,A,B]:-[[Q,A,B,C]])).

  
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
    train_examples(T,Pos,Neg),
    max_time(MaxTime),
    catch(call_with_time_limit(MaxTime,learn(Pos,Neg,Prog)),time_limit_exceeded,(writeln('%timeout'),false)),!,
    pprint(Prog).
learn_aux(_,false).

do_test:-
    tasks(Tasks),
    maplist(do_test,Tasks),
    halt.

do_test(Task):-
    test_examples(Task,Pos),
    (current_predicate(Task/2) ->
        (
            format('%solved,~w,~w\\n',[Task,1]),
            forall(member(X,Pos),(call(X) -> format('~w,~w\\n',[1,1]); format('~w,~w\\n',[1,0])))
        );
        (
            format('%solved,~w,~w\\n',[Task,0]),
            forall(member(_,Pos),format('~w,~w\\n',[0,0]))
        )).

gen_e(Task,Input,Output,Out):-
    Out=..[Task,s(Input,Output),s(_,[])].

play_examples(Task,Pos,[]):-
    findall(X,(play_pos(Task,A,B),gen_e(Task,A,B,X)),Pos1),
    sort_examples(Pos1,Pos).

train_examples(Task,Pos,[]):-
    findall(X,(build_pos(Task,A,B),gen_e(Task,A,B,X)),Pos1),
    sort_examples(Pos1,Pos).

test_examples(Task,Pos):-
    findall(X,(test_pos(Task,A,B),gen_e(Task,A,B,X)),Pos).

add_len(Atom,Len-Atom):-
    Atom=..[_Task,s(Input,_Output),s(_,[])],
    length(Input,Len).
remove_len(_-Atom,Atom).

sort_examples(L1,L2):-
  maplist(add_len,L1,L3),
  keysort(L3,L4),
  maplist(remove_len,L4,L2).
"""

def call_prolog(action,load_files,output):
    cmd = "load_files(['experiment',{}],[silent(true)]). ".format(','.join(load_files))
    cmd += '{}.'.format(action)
    print(cmd)
    with open(output, 'w') as outf:
        p = subprocess.Popen(['swipl','-q','-G8g','-T8g','-L8g'], stdin=subprocess.PIPE, stdout=outf)
        p.stdin.write(cmd.encode())
        (output, err) = p.communicate()

def gen_syn_data(playtime,k):
    call_prolog('b({})'.format(playtime),["'gen-exs'"],'data/play-{}-{}.pl'.format(playtime,k))

def load_probs(nowrites=True):
    dic = {}
    with open('probs.txt', 'r') as f:
        for line in f:
            if line.startswith('# -*- coding: utf-8 -*-'):
                continue
            xs=line.strip().split('=>')
            if len(xs)==1 and len(xs[0])>0:
                try:
                    problem='b'+xs[0][2:]
                except:
                    continue
            elif len(xs)>1:
                if problem not in dic:
                    dic[problem] = []
                dic[problem].append(xs)
    return dic

def gen_data(dic,tasks):
    random.shuffle(tasks)
    print_train=[]
    print_test=[]
    for problem,examples in dic.items():
        if len(examples) < 10:
            continue
        random.shuffle(examples)
        train = examples[:5]
        test = examples[5:10]
        for x in train:
            print_train.append((problem,list(x[0].strip()),list(x[1].strip())))
        for x in test:
            print_test.append((problem,list(x[0].strip()),list(x[1].strip())))

    for problem,a,b in print_train:
        yield 'build_pos({},{},{}).\n'.format(problem,a,b)
    for problem,a,b in print_test:
        yield 'test_pos({},{},{}).\n'.format(problem,a,b)
    yield 'tasks({}).\n'.format(tasks)


def gen_real_data():
    dic=load_probs()
    tasks = list(dic.keys())
    for x in gen_data(dic,tasks):
        yield x

def do_gen_data():
    for k in trials:
        for playtime in playtimes:
            gen_syn_data(playtime,k)
        with open('data/build-{}.pl'.format(k),'w') as f:
            for x in gen_real_data():
                f.write( x + '\n')

def play_and_buid():
    for system in systems:
        for k in trials:
            for p in playtimes:
                playf="'data/play-{}-{}'".format(p,k)
                buildf="'data/build-{}'".format(k)
                programf=f"programs/{system}/{p}-{k}.pl"
                call_prolog('a',[playf,buildf],programf)


def call_prolog_new(action,load_files,output):
    cmd = "load_files([{}],[silent(true)]). ".format(','.join(load_files))
    cmd += '{}.'.format(action)
    with open(output, 'w') as outf:
        p = subprocess.Popen(['swipl','-q','-G8g','-T8g','-L8g'], stdin=subprocess.PIPE, stdout=outf)
        p.stdin.write(cmd.encode())
        print(cmd)
        (output, err) = p.communicate()


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


def build_p():
    print(f"running build_p; trials")
    SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'
    if not os.path.exists("./programs_p"):
        os.mkdir("./programs_p")

    if not os.path.exists(f"./programs_p/{SETUP}"):
        os.mkdir(f"./programs_p/{SETUP}")

    for k in trials:
        for p in playtimes:
            print(f"trial {k} play {p}")
            playf = f"{ROOT_FILE}/refactored_programs_p/strings/{SETUP}/refactored_programs-{p}-{k}.pl" # "'data/play-{}-{}'".format(p,k)
            buildf = f"'{ROOT_FILE}/build/strings/build-{k}.pl'"   # "'data/build-{}'".format(k)
            programf = f"./programs_p/{SETUP}/programs-{p}-{k}.pl"  # f"programs/{s}/programs-{p}-{k}.pl"
            refprimf = f"./programs_p/{SETUP}/refactored_primitives-{p}-{k}.pl"  # f"programs/{s}/programs-{p}-{k}.pl"
            expf = f"./programs_p/{SETUP}/experiment-{p}-{k}.pl"  # f"programs/{s}/programs-{p}-{k}.pl"

            prims = prepare_play_as_background(playf, refprimf)
            generate_experiment_file(expf, refprimf, prims)

            files_to_load = [f"'{expf}'", f"'{refprimf}'", buildf]

            call_prolog_new('b', files_to_load, programf)

def test():
    for system in systems:
        for k in trials:
            for p in playtimes:
                buildf="'data/build-{}'".format(k)
                programf=f"'programs/{system}/{p}-{k}.pl'"
                resultsf=f'results/{system}/{p}-{k}.pl'
                call_prolog('do_test',[buildf,programf],resultsf)

# def read_file(p,k):
#     num_solved=0
#     acc=[]
#     fname = f'results/{system}/{p}-{k}.pl'
#     with open(,'r') as f:
#         for line in f:
#             line=line.strip()
#             xs=line.split(',')
#             if len(xs) <2:
#                 continue
#             if line.startswith('%solved'):
#                 num_solved+=int(xs[2])
#             else:
#                 # k_instances+=1
#                 acc+=[int(xs[1])]
#     return num_solved,acc


def test_new():
    # SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_singl{REJECT_SINGLETONS}'
    SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'
    if not os.path.exists("./results_p"):
        os.mkdir("./results_p")

    if not os.path.exists(f"./results_p/{SETUP}"):
        os.mkdir(f"./results_p/{SETUP}")

    for k in trials:
        for p in playtimes:
            buildf = f"'{ROOT_FILE}/build/strings/build-{k}.pl'"
            programf = f"'./programs_p/{SETUP}/programs-{p}-{k}.pl'"
            resultsf = f'./results_p/{SETUP}/programs-{p}-{k}.pl'
            expf = f"'./programs_p/{SETUP}/experiment-{p}-{k}.pl'"  # f"programs/{s}/programs-{p}-{k}.pl"
            #call_prolog('do_test',[buildf,programf],resultsf)

            call_prolog_new('do_test', [expf, buildf, programf], resultsf)


def get_acc(system,p):
    tasks=['b36', 'b132', 'b246', 'b167', 'b87', 'b304', 'b47', 'b94', 'b284', 'b116', 'b157', 'b239', 'b224', 'b285', 'b215', 'b179', 'b92', 'b227', 'b111', 'b99', 'b35', 'b38', 'b307', 'b91', 'b151', 'b83', 'b61', 'b247', 'b298', 'b67', 'b120', 'b325', 'b63', 'b7', 'b48', 'b33', 'b27', 'b108', 'b78', 'b252', 'b133', 'b1', 'b80', 'b139', 'b100', 'b308', 'b30', 'b136', 'b109', 'b3', 'b103', 'b149', 'b323', 'b29', 'b34', 'b189', 'b293', 'b134', 'b43', 'b314', 'b326', 'b324', 'b188', 'b123', 'b137', 'b98', 'b4', 'b283', 'b300', 'b249', 'b162', 'b24', 'b56', 'b292', 'b241', 'b327', 'b23', 'b6', 'b238', 'b186', 'b81', 'b156', 'b73', 'b102', 'b153', 'b113', 'b37', 'b76', 'b196', 'b5', 'b309', 'b25', 'b184', 'b181']

    all_accs=[]
    accs=[]
    for k in trials:
        k_acc=[]
        fname = f'results/{system}/{p}-{k}.pl'
        with open(fname,'r') as f:
            data=f.read()
            probs=data.split('%')
            for prob in probs:
                xs=prob.split('\n')
                if len(xs) == 0:
                    continue
                if xs[0].startswith('solved'):
                    (_,t,solved) = xs[0].split(',')
                    if t not in tasks:
                        continue
                for vs in xs[1:]:
                    vs = vs.split(',')
                    if len(vs) != 2:
                        continue
                    all_accs.append(int(vs[1]))
                    k_acc.append(int(vs[1]))
        accs.append(np.mean(k_acc))
    return (np.mean(accs),stats.sem(accs),all_accs)


def get_acc_new(setup,p):
    tasks=['b36', 'b132', 'b246', 'b167', 'b87', 'b304', 'b47', 'b94', 'b284', 'b116', 'b157', 'b239', 'b224', 'b285', 'b215', 'b179', 'b92', 'b227', 'b111', 'b99', 'b35', 'b38', 'b307', 'b91', 'b151', 'b83', 'b61', 'b247', 'b298', 'b67', 'b120', 'b325', 'b63', 'b7', 'b48', 'b33', 'b27', 'b108', 'b78', 'b252', 'b133', 'b1', 'b80', 'b139', 'b100', 'b308', 'b30', 'b136', 'b109', 'b3', 'b103', 'b149', 'b323', 'b29', 'b34', 'b189', 'b293', 'b134', 'b43', 'b314', 'b326', 'b324', 'b188', 'b123', 'b137', 'b98', 'b4', 'b283', 'b300', 'b249', 'b162', 'b24', 'b56', 'b292', 'b241', 'b327', 'b23', 'b6', 'b238', 'b186', 'b81', 'b156', 'b73', 'b102', 'b153', 'b113', 'b37', 'b76', 'b196', 'b5', 'b309', 'b25', 'b184', 'b181']

    all_accs=[]
    accs=[]
    for k in trials:
        k_acc=[]
        fname = f'results_p/{setup}/programs-{p}-{k}.pl'
        with open(fname,'r') as f:
            data=f.read()
            probs=data.split('%')
            for prob in probs:
                xs=prob.split('\n')
                if len(xs) == 0:
                    continue
                if xs[0].startswith('solved'):
                    (_,t,solved) = xs[0].split(',')
                    if t not in tasks:
                        continue
                for vs in xs[1:]:
                    vs = vs.split(',')
                    if len(vs) != 2:
                        continue
                    all_accs.append(int(vs[1]))
                    k_acc.append(int(vs[1]))
        print({"system": "knorf", "playtask": p, "trial": k, "accuracy": p.mean(k_acc)})
        accs.append(np.mean(k_acc))
    return (np.mean(accs),stats.sem(accs),all_accs)


def results():
    system_accs = {}
    for system in systems:
        print(system)
        system_accs[system]=[]
        for p in playtimes:
            (acc,sem,all_accs) = get_acc(system,p)
            system_accs[system].extend(all_accs)
            print('({},{}) +- (0,{})'.format(p,round(acc*100,2),round(sem*100,2)))


def results_new():
    system_accs = {}
    for system in ['playgol']:
        print(system)
        SETUP = f'literals{MIN_LITERALS}-{MAX_LITERALS}_layer{MAX_LAYERS}_time{MAX_TIME_S}s_prune{PRUNE}_alt{EXCLUDE_ALTERNATIVES}_rcands{EXCLUDE_REDUNDANT_CANDS}_rr{EXCLUDE_REDUNDANCIES}_mr{MINIMISE_REDUNDANCIES}_singl{REJECT_SINGLETONS}'
        system_accs[system]=[]
        for p in playtimes:
            (acc,sem,all_accs) = get_acc_new(SETUP,p)
            system_accs[system].extend(all_accs)
            #print('({},{}) +- (0,{})'.format(p,round(acc*100,2),round(sem*100,2)))


def parse_size(setup, p, k):
    playf = f"{ROOT_FILE}/refactored_programs_p/strings/{setup}/refactored_programs-{p}-{k}.pl.log"
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
            print({"playtasks": p, "trial": t, "system": "knorf", "theory_size": ref})
            print({"playtasks": p, "trial": t, "system": "metagol", "theory_size": ori})

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
            start_time = datetime.datetime(day=int(tmp[6]), month=int(tmp[7]), year=int(tmp[8]), hour=int(time[0]), minute=int(time[1]), second=int(float(time[2])))
        if '% finished solving build tasks' in line:
            tmp = line.strip().split()
            time = tmp[-1].split(":")
            end_time = datetime.datetime(day=int(tmp[6]), month=int(tmp[7]), year=int(tmp[8]), hour=int(time[0]), minute=int(time[1]), second=int(float(time[2])))

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
    do_gen_data()
if cmd == 'learn':
    play_and_buid()
if cmd == 'learn-build-p':
    build_p()
if cmd == 'test':
    test_new()
if cmd == 'results':
    results_new()
if cmd == 'size':
    theory_sizes()
if cmd == 'runtime':
    runtimes()
if cmd == 'program-size':
    build_program_size()