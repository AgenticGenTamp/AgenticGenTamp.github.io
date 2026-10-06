import sys, numpy as np
from simutil import make_env, obj, robot_q
from approach import GeneratedApproach
N=int(sys.argv[1]); sd=int(sys.argv[2]); env=make_env(N); ap=GeneratedApproach(env.action_space,None,{})
s,info=env.reset(seed=sd,options={'object_count':N}); ap.reset(s,info)
for k in range(int(sys.argv[3])):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
    if k>=int(sys.argv[4]) and k%5==0:
        b,q=robot_q(s); kind,pos,R,g,_,_,_=ap.wps[min(ap.wi,len(ap.wps)-1)]
        print(k, ap.phase, np.round(ap.last_err,2),'q',np.round(q,2),'a',np.round(a[3:10],2),'I',np.round(ap.I,2))
    if term: break
