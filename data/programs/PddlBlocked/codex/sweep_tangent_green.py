from env_client import make_env
from approach import GeneratedApproach
import numpy as np, math, sys

seed=int(sys.argv[1]);e=make_env();hits=[]
for delta in np.arange(-.6,.601,.02):
 s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 for _ in range(80):
  s,*_=e.step(p.get_action(s))
  if p.stage==5:break
 base_theta=p.theta;theta=p.w(base_theta+delta);c,z=math.cos(theta),math.sin(theta)
 p.theta=theta;p.off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]])
 t=p.target(p.green);t[0]=min(5.,t[0])
 for _ in range(12):s,*_=e.step(p.motion(s,t,1,lift=True))
 for _ in range(3):s,*_=e.step(p.motion(s,t,1))
 a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 if p.g(s,'robot','grasp_active')>.5:hits.append((round(delta,3),p.robot(s).tolist()))
e.close();print(seed,hits)
