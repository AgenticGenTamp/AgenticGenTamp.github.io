from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();s,i=E.reset(seed=71);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,i)
for k in range(40):
 a=p.get_action(s);s,r,d,t,inf=E.step(a)
a=p.get_action(s);robot=s.get_object_from_name('robot')
def q(s):return np.array([s.get(robot,f) for f in ['x','y','theta']])
for frac in [1,.75,.5,.25]:
 start=q(s);aa=a.copy();aa[:3]*=frac
 s,*_=E.step(aa);moved=q(s)-start;print('fraction',frac,'change',moved,flush=True)
 if np.linalg.norm(moved)>1e-6:
  aa[:3]=-moved;s,*_=E.step(aa);print('restored',q(s)-start,flush=True)
for dx,dy in [(0,.02),(.02,0),(-.02,0),(0,-.02)]:
 start=q(s);aa=np.array([dx,dy,0,0,1]);s,*_=E.step(aa);moved=q(s)-start;print('translation',dx,dy,'change',moved,flush=True)
 if np.linalg.norm(moved)>1e-6:
  aa[:3]=-moved;s,*_=E.step(aa)
E.close()
