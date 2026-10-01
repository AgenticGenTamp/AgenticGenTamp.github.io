"""Search whether a blocker-grasp posture can transfer directly to green0."""
import math
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))

def move(e,s,p,target,theta,lift=False,steps=28):
    p.theta=theta
    c,z=math.cos(theta),math.sin(theta)
    p.off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]])
    for _ in range(steps): s,*_=e.step(p.motion(s,target,1.,lift=lift))
    return s

def run(seed,theta,delta,lift):
    e=make_env();s,info=e.reset(seed=seed)
    p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
    c,z=math.cos(theta),math.sin(theta)
    off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]])
    bt=p.blocker-off
    # Route around the table before deploying the arm.
    side=1.52 if bt[1]>=0 else -1.52
    for t in ([np.array([3.35,side]),np.array([bt[0],side]),bt] if bt[0]>3.6 else [bt]):
        s=move(e,s,p,t,theta,False)
    a=np.zeros(11,np.float32);a[10]=-1.;s,*_=e.step(a)
    if g(s,'blocker','grasp_active')<.5:e.close();return None
    # Retain blocker in the original grasp posture while carrying it aside.
    s=move(e,s,p,np.clip(bt+delta,[-.9,-1.7],[5.,1.7]),theta,lift)
    a[10]=1.;s,*_=e.step(a)
    if g(s,'blocker','grasp_active')>.5:e.close();return ('refused',)
    gt=p.green-off
    # Return via open space, keeping precisely the posture that grasped blocker.
    side=1.52 if gt[1]>=0 else -1.52
    for t in [np.array([p.robot(s)[0],side]),np.array([gt[0],side]),gt]:
        s=move(e,s,p,t,theta,False)
    a[10]=-1.;s,*_=e.step(a)
    hit=g(s,'green0','grasp_active')>.5
    result=(hit, [g(s,'robot','base_x'),g(s,'robot','base_y')],
            [g(s,'robot','grasp_tf_x'),g(s,'robot','grasp_tf_y'),g(s,'robot','grasp_tf_z')])
    e.close();return result

if __name__ == '__main__':
  seed=int(sys.argv[1]) if len(sys.argv)>1 else 101
  deltas=[np.array(x) for x in [(0,.4),(0,-.4),(.35,0),(-.4,0),(.25,.35),(.25,-.35)]]
  for theta in np.arange(-2.5,.41,.1):
    for delta in deltas:
      for lift in (False,True):
        r=run(seed,float(theta),delta,lift)
        if r and r[0] is True:
          print('HIT',seed,theta,delta.tolist(),lift,r,flush=True);raise SystemExit
  print('MISS',seed)
