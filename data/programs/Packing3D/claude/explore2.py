from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
r = obs.get_object_from_name("robot")
def rob(o):
    return {f: round(float(o.get(r,f)),4) for f in o.type_features[r.type]}
print("init", rob(obs))
# test: apply delta joint_1 = 0.2
a = np.zeros(11); a[3]=0.2
for i in range(3):
    obs, rew, term, trunc, info = env.step(a)
    print(i, rew, term, trunc, rob(obs))
# test base
a = np.zeros(11); a[0]=0.2; a[1]=-0.1; a[2]=0.1
obs, rew, term, trunc, info = env.step(a)
print("base", rob(obs))
# gripper close
a = np.zeros(11); a[10]=-1
obs, rew, term, trunc, info = env.step(a)
print("close", rob(obs))
a = np.zeros(11); a[10]=1
obs, rew, term, trunc, info = env.step(a)
print("open", rob(obs))
# out of bounds action
a = np.zeros(11); a[3]=5.0
obs, rew, term, trunc, info = env.step(a)
print("big", rob(obs))
env.close()
