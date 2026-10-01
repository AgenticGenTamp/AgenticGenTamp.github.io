"""Two-joint grid search for a top-down grasp at a selected isolated cube."""

import argparse
import numpy as np
from env_client import make_env

ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=1); ap.add_argument('--cube',default='cube0'); args=ap.parse_args()
env=make_env(); s,_=env.reset(seed=args.seed); cube=s.get_object_from_name(args.cube)
x,y=s.get(cube,'pose_x'),s.get(cube,'pose_y'); print('target',x,y)
a=np.zeros(11,np.float32);a[1]=y;a[10]=1;s,*_=env.step(a)
# Serpentine grid centered on the empirically smooth radial-reach curve.
guess2=.75+2.0*(x-.431)
guess6=.75+5.5*(x-.431)
d2s=np.linspace(guess2-.16,guess2+.16,17); d6s=np.linspace(guess6-.20,guess6+.20,21)
trial=0
for row,d2 in enumerate(d2s):
  vals=d6s if row%2==0 else d6s[::-1]
  for d6 in vals:
    robot=s.get_object_from_name('robot'); cur2=s.get(robot,'joint_2'); cur6=s.get(robot,'joint_6')
    a=np.zeros(11,np.float32);a[4]=np.clip((-.35+d2)-cur2,-.4,.4);a[8]=np.clip((-.87+d6)-cur6,-.4,.4);a[10]=1
    s,*_=env.step(a)
    a=np.zeros(11,np.float32);a[10]=-1;s,r,term,trunc,_=env.step(a);robot=s.get_object_from_name('robot');trial+=1
    if s.get(robot,'grasp_active')>.5:
      print('GRASP',trial,'d2,d6',d2,d6,'actual',s.get(robot,'joint_2')+.35,s.get(robot,'joint_6')+.87);env.close();raise SystemExit
print('NO_GRASP');env.close()
