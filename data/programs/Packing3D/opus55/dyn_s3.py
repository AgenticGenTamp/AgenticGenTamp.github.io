from env_client import make_env
import numpy as np
env = make_env()
ys=[]
for seed in range(200):
    obs,info=env.reset(seed=seed)
    for n in obs.get_object_names():
        o = obs.get_object_from_name(n) if isinstance(n,str) else n
        if 'part' in str(o): ys.append(abs(float(obs.get(o,'pose_y'))))
ys=np.array(ys); print("min|y|",ys.min(), "hist", np.histogram(ys,bins=[0,0.15,0.2,0.25,0.3,0.35])[0])
print(env.observation_space.type_features)
