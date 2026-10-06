import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import *
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,_=env.reset(seed=3)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,None)
q0=obs[128:135]; print('q0',q0.round(3), 'base', obs[125:128].round(3))
k=ap.kin
qh=k.ik_multi(np.array([1.26,-0.3,np.pi]),q0,np.array([0.934+0.09,0,0.355]),R_front(0.3))
print('qh',qh.round(3),'diff',(qh-q0).round(2))
for by in [-0.3,-0.2,-0.1,0.0]:
  for psi in [0.0,0.3,0.6,0.9]:
    for bx in [1.22,1.26,1.32]:
      q=k.ik_multi(np.array([bx,by,np.pi]),q0,np.array([0.934+0.09,0,0.355]),R_front(psi))
      if q is not None: print(by,psi,bx,'maxdiff',np.abs(q-q0).max().round(2), (q-q0).round(2))
