from env_client import make_env
import numpy as np

e=make_env();s,_=e.reset(seed=0)
r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))))
def read():return [round(float(s.get(r,f)),6) for f in ['x','y','theta','arm_joint','vacuum']]
for _ in range(2):s,*_=e.step(np.array([.05,0,0,0,0],np.float32))
print('before',read())
s,*_=e.step(np.array([.05,.05,.1,.1,1],np.float32));print('collision',read())
e.close()
