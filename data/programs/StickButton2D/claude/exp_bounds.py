from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=1)
r=obs.get_object_from_name("robot")
def st(o,f): return float(o if False else 0)
def show(obs,tag=""):
    r=obs.get_object_from_name("robot"); s=obs.get_object_from_name("stick")
    print(tag,"robot",[round(float(obs.get(r,f)),3) for f in ["x","y","theta","arm_joint","arm_length","vacuum"]],
          "stick",[round(float(obs.get(s,f)),3) for f in ["x","y","theta"]])
show(obs,"init")
# push +x far
for i in range(200):
    obs,rew,term,trunc,info=env.step(np.array([0.05,0,0,0,0],dtype=np.float32))
    if term: print("TERM at +x", i); break
show(obs,"maxx")
for i in range(200):
    obs,rew,term,trunc,info=env.step(np.array([0,0.05,0,0,0],dtype=np.float32))
    if term: print("TERM at +y", i); break
show(obs,"maxy")
for i in range(200):
    obs,rew,term,trunc,info=env.step(np.array([-0.05,-0.05,0,0,0],dtype=np.float32))
    if term: print("TERM", i); break
show(obs,"minxy")
# arm extension limits
for i in range(30):
    obs,rew,term,trunc,info=env.step(np.array([0,0,0,0.1,0],dtype=np.float32))
show(obs,"armout")
for i in range(30):
    obs,rew,term,trunc,info=env.step(np.array([0,0,0,-0.1,0],dtype=np.float32))
show(obs,"armin")
print("trunc",trunc,"steps used")
env.close()
