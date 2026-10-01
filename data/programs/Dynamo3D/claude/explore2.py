from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
tf = env.observation_space.type_features
r = obs.get_object_from_name('robot')
a = np.zeros(11, dtype=np.float32)
for i in range(60):
    obs, rew, term, trunc, info = env.step(a)
    if i%10==0 or term:
        rr = obs.get_object_from_name('robot')
        print(i, round(rew,4), term, trunc, info, [round(float(obs.get(rr,f)),3) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
        for n in sorted(obs.get_object_names()):
            if n=='robot': continue
            o=obs.get_object_from_name(n)
            print("   ",n, [round(float(obs.get(o,f)),3) for f in ['x','y','z','vx','vy','vz']])
    if term or trunc: break
