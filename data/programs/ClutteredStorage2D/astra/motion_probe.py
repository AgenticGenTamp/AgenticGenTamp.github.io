from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=0);r=s.get_objects(E.observation_space.get_type('crv_robot'))[0]
def vals():return [round(s.get(r,f),4) for f in ['x','y','theta','arm_joint','vacuum']]
print(vals())
for a in [[0,0,0,-.1,0]]*8+[[0,0,.19634954,0,0]]*20+[[0,.05,0,0,0]]*40:
 s,rew,done,trunc,info=E.step(a);print(vals(),done,info)
E.close()
