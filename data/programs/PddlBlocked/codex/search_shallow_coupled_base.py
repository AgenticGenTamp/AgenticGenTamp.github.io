"""Search a base detour that allows lowering then insertion into the gap."""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def setup(env, seed):
    s,i=env.reset(seed=seed);p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,i)
    for _ in range(100):
        if p.stage==5: break
        s,*_=env.step(p.get_action(s))
    # Let stock controller reach its lifted target and stall lowering.
    for _ in range(30): s,*_=env.step(p.get_action(s))
    return s,p
def drive(env,s,p,target,q,n=15):
    for _ in range(n):
        a=p.motion(s,target,1)
        for j in range(7):
            d=q[j]-g(s,'robot','joint_'+str(j+1));d=p.w(d) if j in (4,6) else d
            a[3+j]=np.clip(d,-.05,.05)
        s,*_=env.step(a)
    return s

seed=int(sys.argv[1]);env=make_env()
try:
 for dx in (0.,-.05,-.10,-.15,-.20,-.30,-.40):
  for dy in np.arange(-.60,.61,.05):
   s,p=setup(env,seed);goal=p.target(p.green);detour=goal+np.array([dx,dy])
   s=drive(env,s,p,detour,p.Q.copy(),25)
   q=np.array([g(s,'robot','joint_'+str(j+1)) for j in range(7)])
   lowered=abs(q[1]-p.Q[1])<.02
   s=drive(env,s,p,goal,p.Q.copy(),25)
   a=np.zeros(11,np.float32);a[10]=-1;s,*_=env.step(a)
   if g(s,'green0','grasp_active')>.5:
    print('HIT',seed,'dxdy',dx,round(float(dy),3),'detour',detour,'lowered',lowered,'base',p.robot(s),'q',q);break
  else: continue
  break
 else: print('MISS',seed)
finally:env.close()
