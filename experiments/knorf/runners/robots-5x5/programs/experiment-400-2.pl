
:-['../playgol'].
:-['robots-bk'].
:-['../metagol'].

:-['./programs/refactored_primitives-400-2.pl'].



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

prim(latent5_2/2).
prim(latent21_2/2).
prim(latent23_2/2).
prim(latent13_2/2).
prim(latent3_2/2).
prim(latent2_3/2).
prim(latent32_2/2).
prim(latent6_2/2).
prim(latent73_2/2).
prim(latent11_2/2).
prim(latent4_3/2).
prim(latent7_2/2).
prim(latent15_3/2).
prim(latent16_2/2).
prim(latent26_2/2).
prim(latent15_2/2).
prim(latent9_2/2).
prim(latent7_3/2).
prim(latent5_3/2).
prim(latent94_3/2).
prim(latent19_2/2).
prim(latent18_2/2).
prim(latent1_2/2).
prim(latent3_3/2).
prim(latent41_2/2).
prim(latent10_2/2).
prim(latent4_2/2).
prim(latent2_2/2).
prim(p39/2).
prim(p156/2).
prim(p199/2).
prim(p209/2).
prim(p235/2).
prim(p343/2).
prim(p370/2).
prim(p24/2).
prim(p40/2).
prim(p67/2).
prim(p144/2).
prim(p149/2).
prim(p167/2).
prim(p168/2).
prim(p168/2).
prim(p202/2).
prim(p230/2).
prim(p236/2).
prim(p247/2).
prim(p19/2).
prim(p22/2).
prim(p34/2).
prim(p35/2).
prim(p56/2).
prim(p86/2).
prim(p203/2).
prim(p272/2).
prim(p284/2).
prim(p309/2).
prim(p352/2).
prim(p359/2).
prim(p389/2).
prim(p300/2).
prim(p373/2).

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

