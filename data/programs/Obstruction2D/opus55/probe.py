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
print(R(obs), B(obs))
# move right toward block center x=0.595
for i in range(20):
    a = np.array([0.05,0,0,0.1,0],dtype=np.float32)
    obs,r,t,tr,_ = env.step(a)
    print(i, R(obs), r)
# move down
for i in range(20):
    obs,r,t,tr,_ = env.step(np.array([0,-0.05,0,0,1],dtype=np.float32))
    print('down',i,R(obs),B(obs))
for i in range(6):
    obs,r,t,tr,_ = env.step(np.array([0,0.05,0,0,1],dtype=np.float32))
    print('up',i,R(obs),B(obs))
for i in range(3):
    obs,r,t,tr,_ = env.step(np.array([0,0.0,0,0,0],dtype=np.float32))
    print('rel',i,R(obs),B(obs))
