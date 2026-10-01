"""Test Kinova even joints on their equivalent negative-angle branch."""
import numpy as np
from env_client import make_env

env=make_env(); state,_=env.reset(seed=1); robot=state.get_object_from_name('robot'); rod=state.get_object_from_name('cuboid_1')
target=np.array([-3.1003,-2.4,.2638,-.4262,-1.5708,-.1238,2.8392])
def g(o,f):return float(state.get(o,f))
def step(a):
 global state;state,*_=env.step(np.asarray(a,np.float32))
for k in range(180):
 q=np.array([g(robot,f'pos_arm_joint{i}') for i in range(1,8)]);e=target-q;e[[0,2,4,6]]=(e[[0,2,4,6]]+np.pi)%(2*np.pi)-np.pi
 a=np.zeros(11);a[3:10]=np.clip(.8*e,-.1,.1);a[-1]=1;step(a)
print('q',np.round(q,3),flush=True)
orig=np.array([g(rod,f) for f in ('x','y','z')])
for y in np.linspace(orig[1]-.3,orig[1]+.3,13):
 for x in np.linspace(orig[0]-.75,orig[0]-.05,29):
  for _ in range(3):
   e=np.array([x-g(robot,'pos_base_x'),y-g(robot,'pos_base_y')]);a=np.zeros(11);a[:2]=np.clip(e/.87,-.1,.1);a[-1]=1;step(a)
  a=np.zeros(11);a[-1]=0;step(a);a[0]=.02;step(a)
  p=np.array([g(rod,f) for f in ('x','y','z')])
  if np.linalg.norm(p-orig)>.003:print('HIT',x,y,p,flush=True);env.close();raise SystemExit
print('MISS',flush=True);env.close()
