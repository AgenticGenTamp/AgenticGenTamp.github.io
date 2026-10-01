from env_client import make_env
import numpy as np

def dump(e,s):
    for t in e.observation_space.types:
        fs=e.observation_space.type_features[t]
        for o in s.get_objects(t):
            print(o.name, {f:round(float(s.get(o,f)),4) for f in fs if f not in ('color_r','color_g','color_b','z_order','static')})

def robot(e,s):
    o=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))))
    return [round(float(s.get(o,f)),5) for f in ['x','y','theta','arm_joint','arm_length','vacuum']]

for seed in [0,1,2]:
    e=make_env();s,i=e.reset(seed=seed)
    print('SEED',seed,'LIMIT',e.max_steps,'INFO',i);dump(e,s)
    for a in [[0,0,0,-.1,0]]*5 + [[.05,0,0,0,0]]*3+[[0,.05,0,0,0]]*3 + [[0,0,.196,0,0]]*2:
        s,r,t,tr,i=e.step(np.array(a,dtype=np.float32));print(a,robot(e,s),r,t,tr,i)
    e.close()
