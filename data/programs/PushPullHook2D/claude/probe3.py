from env_client import make_env
import numpy as np
env=make_env()
obs,_=env.reset(seed=42)
np.save("obs42.npy",obs)
print(obs.tolist())
env.close()
