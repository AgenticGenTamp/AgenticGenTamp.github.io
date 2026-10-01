"""Search collision-safe blocker headings for the exact due-east pen."""
import math
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))


seed=int(sys.argv[1])
env=make_env()
for theta in np.arange(-math.pi, math.pi+.001, .04):
    s,info=env.reset(seed=seed);p=GeneratedApproach(env.action_space,env.observation_space,{})
    p.reset(s,info);p.theta=float(theta);p.west_branch=False
    c,z=math.cos(theta),math.sin(theta)
    p.off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]])
    t=p.target(p.blocker)
    if not (-1<=t[0]<=5 and -1.75<=t[1]<=1.75): continue
    side=1.6 if t[1]>=0 else -1.6
    p.approach_route=[np.array([3.4,side]),np.array([t[0],side]),t]
    for k in range(65):
        s,*_=env.step(p.get_action(s))
        if p.stage>=2: break
    if g(s,'blocker','grasp_active')>.5:
        print('HIT',round(theta,5),'target',t.round(5),'steps',k,'stage',p.stage)
env.close()
