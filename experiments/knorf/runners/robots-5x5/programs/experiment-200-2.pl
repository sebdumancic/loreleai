
:-['../playgol'].
:-['robots-bk'].
:-['../metagol'].

:-['./programs/refactored_primitives-200-2.pl'].



:- use_module(library(time)).

play_time_interval(2).
max_build_time(60).
max_play_depth(4).
max_build_depth(5).
cpus(4).
max_right(5).
max_forwards(5).



%% tell metagol to use the BK
prim(move_left/2).
prim(move_right/2).
prim(move_forwards/2).
prim(move_backwards/2).
prim(grab_ball/2).
prim(drop_ball/2).

prim(latent17_2/2).
prim(latent9_2/2).
prim(latent3_2/2).
prim(latent10_3/2).
prim(latent2_2/2).
prim(latent1_2/2).
prim(latent5_2/2).
prim(latent7_2/2).
prim(latent6_3/2).
prim(latent6_2/2).
prim(latent15_2/2).
prim(latent1_3/2).
prim(latent3_3/2).
prim(latent13_2/2).
prim(latent7_3/2).
prim(p14/2).
prim(p71/2).
prim(p186/2).
prim(p91/2).
prim(p112/2).
prim(p171/2).
prim(p35/2).
prim(p38/2).
prim(p76/2).
prim(p82/2).
prim(p102/2).
prim(p132/2).
prim(p140/2).

%% metarules
metarule([P/2,Q/2],([P,A,B]:-[[Q,A,B]])).
metarule([P/2,Q/2,R/2],([P,A,B]:-[[Q,A,C],[R,C,B]])).

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
    max_build_depth(BuildDepth),
    playgol:update_depth(BuildDepth),
    max_build_time(BuildTime),
    retractall(max_time(_)),
    assert(max_time(BuildTime)),
    tasks(Tasks),
    learn_tasks(Tasks,Progs),
    length(Progs,N),
    format('% num solved ~w\n',[N]),
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
            format('%solved,~w,~w\n',[Task,1]);
            format('%solved,~w,~w\n',[Task,0])))),
    halt.

