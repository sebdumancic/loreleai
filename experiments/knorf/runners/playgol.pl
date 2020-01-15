:- module(playgol,[playgol/1]).

playgol(Tasks):-
    iterate(Tasks,[],1).

update_depth(Depth):-
    retractall(metagol:max_clauses(_)),
    assert(metagol:max_clauses(Depth)).

update_timeout(Depth):-
    play_time_interval(Interval),
    MaxTime is round((2**(Depth-1) * Interval)),
    retractall(max_time(_)),
    assert(max_time(MaxTime)).

iterate([],_,_Depth):-!.
iterate(_,_,MaxDepth):-
    max_play_depth(MaxDepth),!.
iterate(Tasks1,Seen1,Depth1):-!,
    update_depth(Depth1),
    format('% depth ~w\n',[Depth1]),
    update_timeout(Depth1),
    concurrent_maplist(learn_aux,Tasks1,Xs),!,
    findall(Task,member(false-Task,Xs),Tasks2),
    findall(Prog,member(true-Prog,Xs),Progs1),
    forall(member(Prog,Progs1),(pprint(Prog),metagol:assert_prog(Prog))),
    flatten(Progs1,Clauses1),
    Clauses2 = Clauses1,
    Seen2 = Seen1,
    findall(P/A,member(sub(_,P,A,_,_),Clauses2),NewPrims),
    forall(member(P/A,NewPrims),(
        format('% asserting ~w\n',[P/A]),
        metagol:assert_prim(P/A))),
    succ(Depth1,Depth2),
    iterate(Tasks2,Seen2,Depth2).

learn_aux(T,true-Prog):-
    play_examples(T,Pos,Neg),
    max_time(MaxTime),
    catch(call_with_time_limit(MaxTime,learn(Pos,Neg,Prog)),time_limit_exceeded,false),!.
learn_aux(T,false-T).

dedup_clauses([],Seen,[],Seen).
dedup_clauses([C|T],Seen1,Out,Seen2):-
    code(C,Code),
    memberchk(Code,Seen1),!,
    dedup_clauses(T,Seen1,Out,Seen2).
dedup_clauses([C|T],Seen1,[C|Out],Seen2):-
    code(C,Code),
    dedup_clauses(T,[Code|Seen1],Out,Seen2).

code(sub(Name,_,_,[_|Subs],_),code(Name,Subs)).
