from env_client import make_env
import numpy as np
env = make_env()
obs,info = env.reset(seed=0)
r = obs.get_object_from_name("robot")
def p(o):
    return {f: round(float(o_[0].get(o,f)),4) for f in o_[0].type_features[o.type]}
def show(obs,tag=""):
    r = obs.get_object_from_name("robot")
    print(tag, {f: round(float(obs.get(r,f)),4) for f in ["x","y","theta","arm_joint","arm_length","vacuum"]})
show(obs,"init")
# retract arm fully
for i in range(5):
    obs,rew,t,tr,info = env.step(np.array([0,0,0,-0.1,0.0]))
    show(obs,"retract%d"%i)
# move right
for i in range(6):
    obs,rew,t,tr,info = env.step(np.array([0.05,0,0,0,0.0]))
    show(obs,"right%d"%i)
print("rew",rew,t,tr)
# extend arm
for i in range(5):
    obs,rew,t,tr,info = env.step(np.array([0,0,0,0.1,0.0]))
    show(obs,"ext%d"%i)
for i in range(3):
    obs,rew,t,tr,info = env.step(np.array([0,0,0.19,0,0.0]))
    show(obs,"rot%d"%i)
env.close()
