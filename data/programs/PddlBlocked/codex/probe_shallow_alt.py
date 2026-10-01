"""Try the opposite x-limit tangent after the blocker has been dropped."""
import math, sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def drive(e,s,p,target,q,lift=True,limit=40):
 for _ in range(limit):
  a=np.zeros(11,np.float32); a[:2]=np.clip(target-p.robot(s),-.2,.2)
  a[2]=np.clip(p.w(p.theta-g(s,'robot','base_rot')),-.2,.2)
  for j in range(7):
   d=q[j]-g(s,'robot','joint_'+str(j+1)); d=p.w(d) if j in (4,6) else d
   a[3+j]=np.clip(d,-.2,.2)
  a[10]=1.; old=p.robot(s);s,*_=e.step(a)
  if max(abs(p.robot(s)-target))<.004 and abs(p.w(p.theta-g(s,'robot','base_rot')))<.004 and max(abs(q[j]-g(s,'robot','joint_'+str(j+1))) for j in range(7))<.005:return s,True
 return s,False

for seed in map(int,sys.argv[1:] or [101,191,584,628,760]):
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 for k in range(80):
  if p.stage==5:break
  s,*_=e.step(p.get_action(s))
 # Opposite tangent: base south of green. Move around outside before lowering arm.
 ox=p.green[0]-5.; oy=math.sqrt(max(0.,sum(p.OFF*p.OFF)-ox*ox))
 p.theta=p.w(math.atan2(oy,ox)-math.atan2(p.OFF[1],p.OFF[0]))
 c,z=math.cos(p.theta),math.sin(p.theta);p.off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]])
 qlift=p.Q.copy();qlift[1]-=.2
 targets=[np.array([5.,1.8]),np.array([3.4,1.8]),np.array([3.4,-1.6]),np.array([5.,-1.6]),p.target(p.green)]
 oks=[]
 for t in targets:
  s,ok=drive(e,s,p,t,qlift);oks.append(ok)
 # lower to nominal at target, close
 s,ok=drive(e,s,p,targets[-1],p.Q);oks.append(ok)
 a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 print(seed,'theta',p.theta,'target',targets[-1],'routeok',oks,'grasp',g(s,'green0','grasp_active'),'base',p.robot(s))
 e.close()
