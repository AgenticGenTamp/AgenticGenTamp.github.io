from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
r=env.robot if False else None
rob = obs.get_object_from_name('robot')
for t in range(5):
    a = np.zeros(11, dtype=np.float32)
    a[0]=0.1
    obs, rew, term, trunc, info = env.step(a)
    print(t, rew, term, trunc, info, [round(float(obs.get(rob,f)),3) for f in ['pos_base_x','pos_base_y','pos_base_rot','vel_base_x']])
print(env.max_steps)
env.close()
