from env_client import make_env
from approach import GeneratedApproach
import numpy as np, math, sys

seed=int(sys.argv[1]);e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
def tangent(obj):
 ox=obj[0]-5.;L=np.linalg.norm(p.OFF);oy=-math.sqrt(max(0,L*L-ox*ox));th=math.atan2(oy,ox)-math.atan2(p.OFF[1],p.OFF[0]);c,z=math.cos(th),math.sin(th);off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]]);return th,off,obj-off
th,off,t=tangent(p.blocker);p.theta,p.off=th,off
for w,arm,lift in [(np.array([3.4,1.6]),False,False),(np.array([4.9,1.6]),False,False),(t,True,False)]:
 for _ in range(30):s,*_=e.step(p.motion(s,w,1,lift=lift,arm=arm))
s,*_=e.step(p.motion(s,t,-1))
for _ in range(2):s,*_=e.step(p.motion(s,t,lift=True))
out=t+np.array([0.,.36])
for _ in range(3):s,*_=e.step(p.motion(s,out,lift=True))
s,*_=e.step(p.motion(s,p.robot(s),1,lift=True))
print('dropped',p.robot(s),p.g(s,'robot','grasp_active'))
th,off,t=tangent(p.green);p.theta,p.off=th,off
for _ in range(8):s,*_=e.step(p.motion(s,t,1,lift=True))
print('at lift',p.robot(s),'target',t,p.g(s,'robot','joint_2'))
for _ in range(3):s,*_=e.step(p.motion(s,t,1))
s,*_=e.step(p.motion(s,t,-1))
print('green held',p.g(s,'robot','grasp_active'),'base',p.robot(s),'q2',p.g(s,'robot','joint_2'))
e.close()
