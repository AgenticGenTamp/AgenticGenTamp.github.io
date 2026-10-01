"""Grid probe green grasp after blocker removal for shallow east seeds."""
import math,sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def g(s,n,f):return float(s.get(s.get_object_from_name(n),f))
def run(seed,delta,dy):
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 for _ in range(80):
  if p.stage==5:break
  s,*_=e.step(p.get_action(s))
 q=p.Q.copy()+np.asarray(delta);target=p.target(p.green).copy();target[0]=5.;target[1]+=dy
 for _ in range(30):
  a=np.zeros(11,np.float32);a[:2]=np.clip(target-p.robot(s),-.2,.2)
  a[2]=np.clip(p.w(p.theta-g(s,'robot','base_rot')),-.2,.2)
  for j in range(7):
   d=q[j]-g(s,'robot','joint_'+str(j+1));d=p.w(d) if j in (4,6) else d;a[3+j]=np.clip(d,-.12,.12)
  a[10]=1;s,*_=e.step(a)
 a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 ok=g(s,'green0','grasp_active')>.5
 actual=np.r_[p.robot(s),[g(s,'robot','joint_'+str(j+1)) for j in range(7)]]
 e.close();return ok,actual

def relative(seed):
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);grasp=None
 for _ in range(80):
  old=p.stage;s,*_=e.step(p.get_action(s))
  if p.stage==2 and grasp is None:grasp=p.robot(s).copy()
  if p.stage==5:break
 target=grasp+(p.green-p.blocker)
 for waypoint in (grasp,target):
  for _ in range(20):
   a=p.motion(s,waypoint,1)
   s,*_=e.step(a)
 a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 result=(g(s,'green0','grasp_active')>.5,p.robot(s),target,grasp)
 e.close();return result

for seed in ([101,191] if len(sys.argv)==1 else [int(sys.argv[1])]):
 ok,actual,target,grasp=relative(seed)
 print('REL',seed,ok,'grasp',grasp,'target',target,'actual',actual,flush=True)
 if ok:continue
 for j in range(7):
  for d in (-.4,-.3,-.2,-.1,.1,.2,.3,.4):
   for dy in (-.20,-.10,0,.10,.20):
    delta=np.zeros(7);delta[j]=d;ok,a=run(seed,delta,dy)
    if ok:print('HIT',seed,'joint',j+1,'delta',d,'dy',dy,'actual',a.tolist(),flush=True);raise SystemExit
 print('NONE',seed)
