from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
r = obs.get_objects(env.observation_space.get_type('crv_robot'))[0]
b = obs.get_object_from_name('target_block')
def R(): return [round(float(obs.get(r,f)),4) for f in ['x','y','arm_joint','vacuum']]+[round(float(obs.get(b,f)),4) for f in ['x','y']]
def step(a):
    global obs
    obs,*_ = env.step(np.array(a,dtype=np.float32))
tx = 0.4976+0.097; gy = 0.1+0.159+0.205+0.0015
step([0,0,0,0.1,1])
while abs(R()[0]-tx)>1e-5 or abs(R()[1]-gy)>1e-5:
    dx=tx-R()[0]; dy=gy-R()[1]; s=min(1,0.05/max(abs(dx),abs(dy)))
    step([dx*s,dy*s,0,0,1]); print(R())
step([0,0.05,0,0,1]); print('lift', R())
# lower back and release on same step
step([0,-0.05+0.003,0,0,0]); print('lower+release', R())
step([0,0.05,0,0,0]); print('up', R())
