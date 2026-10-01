import numpy as np
from env_client import make_env
env = make_env()
obs, info = env.reset(seed=0)
obs = np.asarray(obs, dtype=float)
print("obs shape", obs.shape)
np.set_printoptions(precision=4, suppress=True)
for name, sl in [("A",(0,16)),("B",(16,32)),("C",(32,48))]:
    print(name, obs[sl[0]:sl[1]])
print("48:93", obs[48:93])
print("robot 93:115", obs[93:115])
print("info", info)
env.close()
