left(world(X1,Size,Arr),world(X2,Size,Arr)):-
  X1 > 1,
  X2 is X1-1.

right(world(X1,Size,Arr),world(X2,Size,Arr)):-
  X1 < Size,
  X2 is X1+1.

at_start(world(1,_,_)).
not_at_start(A):-
  \+ at_start(A).

at_end(world(Size,Size,_)).
not_at_end(A):-
  \+ at_end(A).

go_start(world(_,Size,Arr),world(1,Size,Arr)).

go_end(world(_,Size,Arr),world(Size,Size,Arr)).

place1(world(X,Size,Arr1),world(X,Size,Arr2)):-
  Dummy =.. [dummy|Arr1],
  arg(X,Dummy,V1),
  V1 < Size,
  succ(V1,V2),
  setarg(X,Dummy,V2),
  Dummy =..[dummy|Arr2].

%% f(A,B):-f1(A,C),place1(C,B).
%% f1(A,B):-right(A,B),at_end(B).
%% f1(A,B):-not(at_end(A)),right(A,C),f1(C,B).

%% :-
%%   A = world(1,[0,0,0,0,0]),
%%   f(A,X),
%%   writeln(X).
%%   %% place1(A,C),
%%   %% place1(C,D),
%%   %% writeln(D).


