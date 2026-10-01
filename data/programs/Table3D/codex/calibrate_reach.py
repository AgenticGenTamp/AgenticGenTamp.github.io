"""Find the shared joint-2/joint-6 offset that reaches each cube x coordinate."""

import argparse
import numpy as np
from env_client import make_env


ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--cube',default='cube0'); args=ap.parse_args()
env=make_env(); s,_=env.reset(seed=args.seed); cube=s.get_object_from_name(args.cube); robot=s.get_object_from_name('robot')
x=s.get(cube,'pose_x'); y=s.get(cube,'pose_y'); print('target',args.cube,x,y)
# Move laterally first and keep the jaw open.
a=np.zeros(11,np.float32); a[1]=y; a[10]=1; s,*_=env.step(a)
base=np.array([0.,-.35,-np.pi,-2.5,0.,-.87,np.pi/2])
for d in np.linspace(.35,2.7,48):
    target=base.copy(); target[1]+=d; target[5]+=d
    robot=s.get_object_from_name('robot')
    for _ in range(8):
        cur=np.array([s.get(robot,f'joint_{i}') for i in range(1,8)])
        a=np.zeros(11,np.float32); a[3:10]=np.clip(target-cur,-.4,.4); a[10]=1
        s,*rest=env.step(a); robot=s.get_object_from_name('robot')
        if np.max(np.abs(target-cur))<.01: break
    a=np.zeros(11,np.float32); a[10]=-1; s,r,term,trunc,_=env.step(a); robot=s.get_object_from_name('robot')
    if s.get(robot,'grasp_active')>.5:
        print('GRASP d',round(float(d),4),'actual',s.get(robot,'joint_2')+.35); break
else: print('NO_GRASP')
env.close()
