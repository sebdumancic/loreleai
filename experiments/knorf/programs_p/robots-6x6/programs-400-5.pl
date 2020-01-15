p24(A,B):-move_left(A,C),p24_1(C,B).
p24_1(A,B):-move_left(A,C),move_backwards(C,B).
p2(A,B):-p24(A,C),p2_1(C,B).
p2_1(A,B):-drop_ball(A,C),move_right(C,B).
p25(A,B):-move_left(A,C),p25_1(C,B).
p25_1(A,B):-p25_2(A,C),p25_2(C,B).
p25_2(A,B):-move_left(A,C),move_forwards(C,B).
