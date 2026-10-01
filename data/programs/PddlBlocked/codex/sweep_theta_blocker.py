from env_client import make_env
from approach import GeneratedApproach
import numpy as np, math

for seed in [11,30,34]:
 hits=[]
 for th in np.linspace(-math.pi,math.pi,25,endpoint=False):
  e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
  p.theta=float(th);c,z=math.cos(th),math.sin(th);p.off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]])
  t=p.target(p.blocker); route=[]
  if t[0]>3.65:
   side=1.7 if t[1]>=0 else -1.7;route=[np.array([3.3,side]),np.array([min(t[0],4.95),side])]
  route.append(t)
  for ri,x in enumerate(route):
   for k in range(25):s,*_=e.step(p.motion(s,x,1,arm=(ri==len(route)-1)))
  a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
  if p.g(s,'robot','grasp_active'):hits.append(round(th,3))
  e.close()
 print(seed,hits)
