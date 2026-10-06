import sys, numpy as np
from simutil import make_env, obj
from approach import GeneratedApproach, quat_R
N=int(sys.argv[1]); sd=int(sys.argv[2]); T=int(sys.argv[3]); ci=int(sys.argv[4])
env=make_env(N); ap=GeneratedApproach(env.action_space,None,{})
s,info=env.reset(seed=sd,options={'object_count':N}); ap.reset(s,info)
for k in range(T):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
    Y=obj(s,'bin_yellow_0'); Rb=quat_R(*Y[3:]); c=obj(s,'cube_%d'%ci)
    l=Rb.T@(c[:3]-Y[:3])
    if k%3==0 or abs(l[1])>0.125: print(k,ap.phase,np.round(l,3), 'tilt', round(np.degrees(np.arccos(np.clip(Rb[2,2],-1,1)))))
    if abs(l[1])>0.13: break
