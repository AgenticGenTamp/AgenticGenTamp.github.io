import sys, numpy as np
from simutil import make_env
from approach import GeneratedApproach
N=int(sys.argv[1]); sd=int(sys.argv[2])
env=make_env(N); ap=GeneratedApproach(env.action_space,None,{})
s,info=env.reset(seed=sd,options={'object_count':N}); ap.reset(s,info)
last=None; out=[]
for k in range(int(sys.argv[3])):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
    if ap.phase!=last: out.append('%s@%d'%(ap.phase,k)); last=ap.phase
    if term: break
print(k,term,' '.join(out))
