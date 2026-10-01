"""Keep the blocker-grasp posture low and transfer it through the open gap."""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def move(e,s,p,target,grip,n=30):
 for _ in range(n): s,*_=e.step(p.motion(s,target,grip,arm=True))
 return s

for seed in map(int,sys.argv[1:] or [101,191,3533,4022]):
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 for _ in range(100):
  if p.stage==2:break
  s,*_=e.step(p.get_action(s))
 grasp_base=p.robot(s).copy();delta=p.green-p.blocker
 # Pull the blocker straight outward without lifting, release, and insert
 # the unchanged grasp posture along exactly the same gap centerline.
 away=grasp_base+.45*p.out
 s=move(e,s,p,away,0.,30)
 a=np.zeros(11,np.float32);a[10]=1.;s,*_=e.step(a)
 released=g(s,'blocker','grasp_active')<.5
 s=move(e,s,p,grasp_base,1.,30)
 s=move(e,s,p,grasp_base+delta,1.,30)
 a[10]=-1.;s,*_=e.step(a)
 print(seed,'released',released,'hit',g(s,'green0','grasp_active'),
       'bases',grasp_base,away,p.robot(s),'desired',grasp_base+delta)
 e.close()
