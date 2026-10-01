"""Find the shortest suffix of the discovered random waypoint grasp route."""
import math, sys
import numpy as np
from env_client import make_env
from probe_grasp_structured import command, val

lo=np.array([0.,-.35,-math.pi,-2.5,0.,-.87,math.pi/2])
hi=lo+np.array([5.2,2.41,2.058,2.66,5.2,2.23,6.77])
rng=np.random.default_rng(0); route=[]
for attempt in range(28):
 q=rng.uniform(lo,hi);radius=rng.uniform(.05,.95);angle=rng.uniform(-math.pi,math.pi)
 route.append((q,radius,angle,rng.uniform(-math.pi,math.pi)))

keep=int(sys.argv[1]) if len(sys.argv)>1 else 2
seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
env=make_env();state,_=env.reset(seed=seed);tx=val(state,'box0','pose_x');ty=val(state,'box0','pose_y')
for q,r,ang,rot in route[-keep:]:
 base=[tx+r*math.cos(ang),ty+r*math.sin(ang),rot]
 state=command(env,state,base,q,1.,repeats=35)
 state=command(env,state,base,q,-1.,repeats=2)
 if val(state,'robot','grasp_active')>.5:
  print('HIT seed',seed,'keep',keep,'q',[val(state,'robot','joint_%d'%i) for i in range(1,8)],'base',[val(state,'robot',f) for f in ('pos_base_x','pos_base_y','pos_base_rot')]);break
else:print('MISS seed',seed,'keep',keep)
env.close()
