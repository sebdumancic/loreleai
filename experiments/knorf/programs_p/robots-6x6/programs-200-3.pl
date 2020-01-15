p60(A,B):-move_right(A,C),p60_1(C,B).
p60_1(A,B):-move_right(A,C),move_right(C,B).
p78(A,B):-move_right(A,C),p78_1(C,B).
p78_1(A,B):-move_right(A,C),move_backwards(C,B).
p105(A,B):-move_right(A,C),p105_1(C,B).
p105_1(A,B):-move_right(A,C),move_forwards(C,B).
p37(A,B):-move_backwards(A,C),p37_1(C,B).
p37_1(A,B):-p78_1(A,C),p78(C,B).
