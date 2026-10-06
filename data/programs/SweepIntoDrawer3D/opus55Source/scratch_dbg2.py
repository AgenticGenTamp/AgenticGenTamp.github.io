import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
import approach as A
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=0)
ap=A.GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
orig=ap.cart_move
def cm(pos,R,**kw):
    p,Ra=ap.ee(); print('cart_move start ee',p.round(3),'target',np.round(pos,3),'roterr',np.linalg.norm(A.rot_err(Ra,R)).round(3),'tilt',getattr(ap,'tilt',None))
    b=kw.get('base_t'); print(' base',ap.base(), b)
    yield from orig(pos,R,**kw)
ap.cart_move=cm
for i in range(265):
    a=ap.get_action(obs); obs,*_=env.step(a)
