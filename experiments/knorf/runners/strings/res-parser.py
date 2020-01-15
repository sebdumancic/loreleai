# for p in range(0,4200,200):
#     pscores=[]
#     for k in range(1,11):
#         kscores=[]
#         with open('results-{}-{}.pl'.format(p,k)) as f:
#             for line in f:
#                 line=line.strip()
#                 xs=line.split(',')
#                 if len(xs) != 2:
#                     continue
#                 k,v = int(xs[0]),int(xs[1])
#                 # if k  == 1:
#                 kscores.append(v)
#         # print(np.mean(kscores))
#         pscores.append(np.mean(kscores))
#     print(p,np.mean(pscores))



system='playgol'
for p in range(0,2200,200):
    for k in range(1,11):
        k_acc=[]
        fname='results/{}/{}-{}.pl'.format(system,p,k)
        with open(fname,'r') as f:
            data=f.read()
            probs=data.split('%')
            for prob in probs:
                xs=prob.split('\n')
                if len(xs) == 0:
                    continue
                if xs[0].startswith('solved'):
                    (_,t,solved) = xs[0].split(',')
                    if t == 'b224' and int(solved) == 1:
                        print(p,k,t,solved)


# counts={}
# for k in [1,2,3,4,5]:
#     for p in range(0,2200,200):
#     # for p in [0]:
#         with open('results/results-{}-{}.pl'.format(p,k)) as f:
#             for line in f:
#                 line = line.strip()
#                 if len(line) == 0:
#                     continue
#                 if line[0] != '%':
#                     continue
#                 xs=line.split(',')
#                 if int(xs[2]) == 1:
#                     name=xs[1]
#                     if name in counts:
#                         counts[name]+=1
#                     else:
#                         counts[name]=1

# # fails = []
# good = []
# for i in range(500):
#     k='b'+str(i)
#     if k in counts:
#         # print(k,counts[k])
#         good.append(k)
#         pass
#     else:
#         pass
#         # print(k,0)
#         # fails.append(k)

# print(len(good))
# print(good)
