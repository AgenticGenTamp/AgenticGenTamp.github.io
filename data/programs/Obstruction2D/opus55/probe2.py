from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
def R(o):
    r = o.get_objects(env.observation_space.get_type('crv_robot'))[0]
    return {f: round(float(o.get(r,f)),4) for f in ['x','y','theta','arm_joint','vacuum']}
def B(o, name='target_block'):
    b = o.get_object_from_name(name)
    return {f: round(float(o.get(b,f)),4) for f in ['x','y','theta']}
def step(a):
    global obs
    obs,r,t,tr,_ = env.step(np.array(a,dtype=np.float32)); return t
tx = 0.4976+0.097
while abs(R(obs)['x']-tx)>1e-4:
    step([np.clip(tx-R(obs)['x'],-.05,.05),0,0,0.1,0])
print(R(obs))
prev=None
for i in range(200):
    step([0,-0.005,0,0,0])
    if R(obs)['y']==prev: break
    prev=R(obs)['y']
print('lowest', R(obs), B(obs))
step([0,0,0,0,1]); print('vac on', R(obs), B(obs))
for i in range(5):
    step([0,0.05,0,0,1]); print('up', R(obs), B(obs))
step([0.05,0,0.1,0,1]); print('rot', R(obs), B(obs))
step([0,0,0,-0.05,1]); print('retract', R(obs), B(obs))
step([0,0,0,0,0]); print('rel', R(obs), B(obs))
step([0,0.05,0,0,0]); print('up', R(obs), B(obs))
