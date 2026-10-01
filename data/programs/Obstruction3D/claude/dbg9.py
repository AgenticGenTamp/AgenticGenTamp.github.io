import numpy as np
from env_client import make_env
from approach import GeneratedApproach
import robot
from s2 import *
env=make_env(); obs,info=env.reset(seed=4, options={"object_count":6})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for i in range(200):
    a=ap.get_action(obs)
    q0=robot_q(obs)
    obs,r,t,tr,info=env.step(a)
    if i>=50 and i<95:
        q1=robot_q(obs)
        print(i,ap.stage,"esc",ap.escape,"md",round(ap.mdscale,3),"|a|",round(np.abs(a[3:10]).max(),4),
              "acc",np.max(np.abs(q1-q0))>1e-9,"esck",ap.esc_k,"hold",int(ap._holding(obs)),"z",round(ap._tool(obs)[2,3],4))
    if t: print("TERM",i); break
