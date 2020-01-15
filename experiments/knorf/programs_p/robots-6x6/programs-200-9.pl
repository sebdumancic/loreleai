p82(A,B):-move_left(A,C),move_right(C,B).
p100(A,B):-p100_1(A,C),p100_1(C,B).
p100_1(A,B):-move_left(A,C),move_forwards(C,B).
