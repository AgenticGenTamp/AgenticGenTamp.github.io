import numpy as np, sys
sys.path.insert(0,'.')
from env_client import make_env
from ctrl import robot_state, action, cubes
from fk import ik, fk
RD = np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
env=make_env(); obs,info=env.reset(seed=0)
b,q,g=robot_state(obs)
qt = ik(np.array([0.45,0.0,-0.25]), RD, q)
print('qt',np.round(qt,3), 'fk', np.round(fk(qt)[:3,3],4))
b0=b.copy()
for t in range(200):
    a=action(b,q,b0,qt,0.0)
    obs,r,te,tr,i=env.step(a); b,q,g=robot_state(obs)
    if t%20==0 or t==199: print(t, np.round(q-qt,3), 'base',np.round(b,3))
print('cubes', {k:np.round(v,3) for k,v in cubes(obs).items()})
env.close()
