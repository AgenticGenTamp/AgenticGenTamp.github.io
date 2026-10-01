from env_client import make_env
import numpy as np

def r(e,s):
 o=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))));return [float(s.get(o,f)) for f in ['x','y','theta','arm_joint','arm_length']]
def step(e,a):
 s,*_=e.step(np.array(a,dtype=np.float32));return s
for seed in [0,1]:
 e=make_env();s,_=e.reset(seed=seed)
 print('seed',seed)
 for j in range(5):
  s=step(e,[0,0,0,.1,0]); print('extend',r(e,s))
 for j in range(8):
  s=step(e,[0,0,0,-.1,0]);print('retract',r(e,s))
 for j in range(12):
  s=step(e,[-.05,0,0,0,0]); print('left',r(e,s))
 e.close()
# initial position, rotate toward zero, then advance into wall. Distinguishes arm endpoint.
for theta in [0.,1.570796,3.141592]:
 e=make_env();s,_=e.reset(seed=0)
 for j in range(35):
  z=r(e,s);delta=(theta-z[2]+np.pi)%(2*np.pi)-np.pi
  s=step(e,[0,0,np.clip(delta,-.196,.196),-.1,0])
 for j in range(35):
  s=step(e,[.005,0,0,-.1,0])
 print('angle wall',theta,r(e,s));e.close()
