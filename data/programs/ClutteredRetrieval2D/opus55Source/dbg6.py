import sys, numpy as np
from test_approach import make
import approach
from approach import *
env=make(); s,info=env.reset(seed=1, options={'object_count':30})
ap=GeneratedApproach(env.action_space, None, {}); ap.reset(s,info)
cap={}
orig=ap._plan_to_any
def w(chk,start,goals,tb,*a,**k):
    res=orig(chk,start,goals,tb,*a,**k)
    if res is None and chk.held: cap['chk']=chk; cap['q']=np.array(start)
    return res
ap._plan_to_any=w
for t in range(200):
    a=ap.get_action(s); s,_,_,_,_=env.step(a)
    if 'chk' in cap: break
chk=cap['chk']; q0=cap['q']; print('q0',q0, chk.free(q0), chk.free(q0,0.0))
h,ok=chk.colliders(q0,margin=0.003); print('coll',[chk.obs.names[i] for i in np.nonzero(h)[0]])
u=np.array([np.cos(q0[2]),np.sin(q0[2])])
for b in np.arange(0.01,0.3,0.02):
    q=q0.copy(); q[:2]-=u*b; print(round(b,2), chk.free(q,0.0), chk.free(q), end='; ')
print()
for daj in [-0.02,-0.05,-0.1]:
    q=q0.copy(); q[3]+=daj; print('aj',daj, chk.free(q,0.0), chk.free(q))
for b in [0.05,0.11,0.19]:
    q=q0.copy(); q[:2]-=u*b; h,ok=chk.colliders(q,margin=0.0); print(b,[chk.obs.names[i] for i in np.nonzero(h)[0]],ok)
    R=chk.robot.bodies(q); 
r,rects=ap._parse(s); m=ap._robot_model(r)
print('held now', ap._held(r,rects,m))
