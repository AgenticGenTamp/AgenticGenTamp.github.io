from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
r = obs.get_object_from_name('robot')
def rs(o):
    return np.round(o.data[r],4).tolist()
print("init", rs(obs))
a = np.zeros(11, dtype=np.float32)
for i in range(5):
    obs, rew, term, trunc, info = env.step(a)
    print(i, rew, term, rs(obs))
# now push joint1 by +0.1
a = np.zeros(11, dtype=np.float32); a[3]=0.1
for i in range(5):
    obs, rew, term, trunc, info = env.step(a)
    print('j1', i, rew, rs(obs))
a = np.zeros(11, dtype=np.float32)
for i in range(5):
    obs, rew, term, trunc, info = env.step(a)
    print('z', i, rew, rs(obs))
a = np.zeros(11, dtype=np.float32); a[0]=0.1; a[1]=0.1; a[2]=0.1
for i in range(5):
    obs, rew, term, trunc, info = env.step(a)
    print('base', i, rew, rs(obs))
env.close()
