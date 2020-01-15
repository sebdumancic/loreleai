%% background knowledge
grab_ball(world(Pos,Pos,false),world(Pos,Pos,true)).

drop_ball(world(Pos,Pos,true),world(Pos,Pos,false)).

move_left(world(X1/Y1,Bpos,false),world(X2/Y1,Bpos,false)):-
    nonvar(X1),
    X1 > 0,
    X2 is X1-1.

move_left(world(X1/Y1,_,true),world(X2/Y1,X2/Y1,true)):-
    nonvar(X1),
    X1 > 0,
    X2 is X1-1.

move_right(world(X1/Y1,Bpos,false),world(X2/Y1,Bpos,false)):-
    nonvar(X1),
    max_right(MAXRIGHT),
    X1 < MAXRIGHT,
    X2 is X1+1.

move_right(world(X1/Y1,_,true),world(X2/Y1,X2/Y1,true)):-
    nonvar(X1),
    max_right(MAXRIGHT),
    X1 < MAXRIGHT,
    X2 is X1+1.

move_backwards(world(X1/Y1,Bpos,false),world(X1/Y2,Bpos,false)):-
  nonvar(Y1),
  Y1 > 0,
  Y2 is Y1-1.

move_backwards(world(X1/Y1,_,true),world(X1/Y2,X1/Y2,true)):-
    nonvar(Y1),
    Y1 > 0,
    Y2 is Y1-1.

move_forwards(world(X1/Y1,Bpos,false),world(X1/Y2,Bpos,false)):-
    nonvar(Y1),
    max_forwards(MAXFORWARDS),
    Y1 < MAXFORWARDS,
    Y2 is Y1+1.

move_forwards(world(X1/Y1,_,true),world(X1/Y2,X1/Y2,true)):-
    nonvar(Y1),
    max_forwards(MAXFORWARDS),
    Y1 < MAXFORWARDS,
    Y2 is Y1+1.