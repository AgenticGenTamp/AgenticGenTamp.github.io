import numpy as np
from env_client import make_env
import approach as A
env=make_env()
orig_fb = A.GeneratedApproach.fallback_plan
cnt={'fb':0,'greedy':0,'none':0}
def fb(self,*a,**k):
    cnt['fb']+=1; return orig_fb(self,*a,**k)
A.GeneratedApproach.fallback_plan=fb
orig_pf = A.GeneratedApproach.compute_plan
def pf(self,*a,**k):
    cnt['greedy']+=1; return orig_pf(self,*a,**k)
A.GeneratedApproach.compute_plan=pf
ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
bad=[]
for oc in [0,1,2,3,4]:
    for seed in range(200,260):
        obs,info=env.reset(seed=seed, options={'object_count':oc}); ap.reset(obs,info)
        for t in range(1000):
            a=ap.get_action(obs); obs,r,term,tr,info=env.step(a)
            if term: break
        if not term: bad.append((oc,seed))
print(cnt, "fails", bad)
env.close()
