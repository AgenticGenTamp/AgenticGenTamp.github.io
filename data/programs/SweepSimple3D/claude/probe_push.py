# Push the base straight forward (-y) through the cube field, log rewards & cube pos
from env_client import make_env
import numpy as np
env = make_env(); os_=env.observation_space; T=os_.get_type
obs,info = env.reset(seed=1)
r=obs.get_objects(T("mujoco_tidybot_robot"))[0]
def rob(o,f): return float(o.get(o.get_object_from_name("robot"),f)) if False else 0
tot=0
for i in range(120):
    a=np.zeros(11,dtype=np.float32); a[1]=-0.1
    obs,rew,term,trunc,info=env.step(a)
    tot+=rew
    if i%10==0 or rew>-0.009:
        rr=obs.get_object_from_name("robot")
        c=obs.get_object_from_name("cube_0")
        print(i, round(rew,4), "base",round(float(obs.get(rr,"pos_base_x")),3),round(float(obs.get(rr,"pos_base_y")),3),
              "cube",round(float(obs.get(c,"x")),3),round(float(obs.get(c,"y")),3),round(float(obs.get(c,"z")),3), term)
    if term: break
print("total",tot)
env.close()
