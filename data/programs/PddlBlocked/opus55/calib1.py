from env_client import make_env
import numpy as np, sys
from kin import *
env = make_env()
obs,_=env.reset(seed=0)
T={t.name:t for t in env.observation_space.type_features}
r=obs.get_objects(T['robot'])[0]
q=np.array([obs.get(r,f) for f in ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7']])
p,R=fk_world((2.5,0,0),q)
print('init tool',p.round(3)); print(R.round(3))
