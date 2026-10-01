"""Search shoulder/elbow/wrist offsets for a far-edge cube."""

import argparse
import numpy as np
from env_client import make_env

ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1);ap.add_argument('--cube',default='cube1');args=ap.parse_args()
env=make_env();s,_=env.reset(seed=args.seed);cube=s.get_object_from_name(args.cube);rng=np.random.default_rng(41)
y=s.get(cube,'pose_y');a=np.zeros(11,np.float32);a[1]=y;a[10]=1;s,*_=env.step(a)
base=np.array([0.,-.35,-np.pi,-2.5,0.,-.87,np.pi/2])
for trial in range(190):
    # Bias around a mostly extended top-down pose, while searching elbow compensation.
    d2=rng.uniform(.8,1.8); d4=rng.uniform(0.,1.3); d6=rng.uniform(1.3,2.7)
    target=base.copy();target[[1,3,5]] += [d2,d4,d6]
    for _ in range(4):
      robot=s.get_object_from_name('robot');cur=np.array([s.get(robot,f'joint_{i}') for i in range(1,8)])
      a=np.zeros(11,np.float32);a[3:10]=np.clip(target-cur,-.4,.4);a[10]=1;s,*_=env.step(a)
    a=np.zeros(11,np.float32);a[10]=-1;s,*_=env.step(a);robot=s.get_object_from_name('robot')
    if s.get(robot,'grasp_active')>.5:
      print('GRASP',trial,'offsets',d2,d4,d6,'actual',[s.get(robot,f'joint_{i}')-base[i-1] for i in [2,4,6]]);break
else:print('NO_GRASP')
env.close()
