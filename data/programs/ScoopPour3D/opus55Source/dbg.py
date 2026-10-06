import sys, numpy as np
from simutil import make_env, obj
from approach import GeneratedApproach
N=10; env=make_env(N); ap=GeneratedApproach(env.action_space,None,{})
s,info=env.reset(seed=0,options={'object_count':N}); ap.reset(s,info)
for k in range(400):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
    if term: break
print(k, term, 'green',np.round(obj(s,'bin_green_0'),3),'yellow',np.round(obj(s,'bin_yellow_0'),3))
for i in range(N): print(np.round(obj(s,'cube_%d'%i)[:3],3))
