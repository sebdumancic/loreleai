p6(A,B):-grab_ball(A,C),p6_1(C,B).
p6_1(A,B):-move_left(A,C),move_forwards(C,B).
p88(A,B):-move_right(A,C),p88_1(C,B).
p88_1(A,B):-p88_2(A,C),p88_2(C,B).
p88_2(A,B):-move_right(A,C),move_backwards(C,B).
