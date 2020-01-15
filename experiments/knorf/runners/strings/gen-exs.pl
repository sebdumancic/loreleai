:-['string-bk'].

max_string_size(20).

prim(copy1/2).
prim(skip1/2).
prim(mk_uppercase/2).
prim(mk_lowercase/2).

rand_prim(P/A):-
    findall(X,prim(X/A),Prims),
    %% random_member(P,Prims).
    member(P,Prims).

%% action(A,B,Action):-
    %% (maybe -> action1(A,B,Action); action2(A,B,Action)).

action(A,B,[P]):-
    rand_prim(P/2),
    Goal=..[P,A,B],
    call(Goal).

f(1,A,B,Prog1,[Action|Prog1]):-!,
    action(A,B,Action).
f(N,A,B,Prog1,[Action|Prog2]):-
    action(A,C,Action),
    succ(M,N),
    f(M,C,B,Prog1,Prog2).

a(In,Out,Prog):-
    max_string_size(MaxSize),
    random_between(3,MaxSize,InputSize),
    A=s(In,Out),
    B=s(_,[]),
    length(In,InputSize),
    bind(In),
    random_between(1,10,ProgramSize),
    f(ProgramSize,A,B,[],Prog).

b(Playtime):-
    forall(between(1,Playtime,I),
        (
        findall(A/B/Prog,a(A,B,Prog),D),
        random_member(A/B/Prog,D),
        format('% ~w\n',[Prog]),
        format('play_pos(p~w,~q,~q).\n',[I,A,B])
    )),
    findall(Y,(between(1,Playtime,I),atom_string(I,X),string_concat("p",X,Y)),Ys),
    format('games(~w).\n',[Ys]).

bind([]).
bind([H|T]):-
    rand_symbol(H),
    bind(T).

rand_symbol(A):-
    findall(X,symbol(X),Xs),
    random_member(A,Xs).

symbol('A').
symbol('B').
symbol('C').
symbol('D').
symbol('E').
symbol('F').
symbol('G').
symbol('H').
symbol('I').
symbol('J').
symbol('K').
symbol('L').
symbol('M').
symbol('N').
symbol('O').
symbol('P').
symbol('Q').
symbol('R').
symbol('S').
symbol('T').
symbol('U').
symbol('V').
symbol('W').
symbol('X').
symbol('Y').
symbol('Z').

symbol('a').
symbol('b').
symbol('c').
symbol('d').
symbol('e').
symbol('f').
symbol('g').
symbol('h').
symbol('i').
symbol('j').
symbol('k').
symbol('l').
symbol('m').
symbol('n').
symbol('o').
symbol('p').
symbol('q').
symbol('r').
symbol('s').
symbol('t').
symbol('u').
symbol('v').
symbol('w').
symbol('x').
symbol('y').
symbol('z').

symbol('0').
symbol('1').
symbol('2').
symbol('3').
symbol('4').
symbol('5').
symbol('6').
symbol('7').
symbol('8').
symbol('9').


symbol(',').
symbol('.').
symbol('@').
symbol('/').
symbol('-').
symbol(':').
symbol(';').
symbol('\\').
symbol('_').
symbol('#').
symbol('<').
symbol('>').
symbol(' ').
symbol('+').
symbol('(').
symbol(')').
symbol('"').
symbol(' ').