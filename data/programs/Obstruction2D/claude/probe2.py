from env_client import make_env
import numpy as np
env = make_env()
def dump(obs, names=None):
    out={}
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        out[name]={f: round(float(obs.get(o,f)),4) for f in obs.type_features[o.type]}
    return out
obs,info=env.reset(seed=42)
r=dump(obs)['robot']; print("init",r)
# move right
for i in range(5):
    obs,rew,t,tr,info=env.step(np.array([0.05,0,0,0,0]))
print("after 5 dx=+0.05", dump(obs)['robot'], rew,t)
for i in range(30):
    obs,rew,t,tr,info=env.step(np.array([0.05,0,0,0,0]))
print("after 35 right", dump(obs)['robot'])
for i in range(10):
    obs,rew,t,tr,info=env.step(np.array([0,-0.05,0,0,0]))
print("after 10 down", dump(obs)['robot'])
for i in range(60):
    obs,rew,t,tr,info=env.step(np.array([0,-0.05,0,0,0]))
print("after down max", dump(obs)['robot'])
for i in range(5):
    obs,rew,t,tr,info=env.step(np.array([0,0,0.19,0,0]))
print("after rot", dump(obs)['robot'])
for i in range(5):
    obs,rew,t,tr,info=env.step(np.array([0,0,0,0.1,0]))
print("after arm out", dump(obs)['robot'])
env.close()
