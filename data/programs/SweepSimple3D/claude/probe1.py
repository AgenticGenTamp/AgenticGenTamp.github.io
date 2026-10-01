from env_client import make_env
import numpy as np
env = make_env()
os_ = env.observation_space
def T(n): return os_.get_type(n)
obs, info = env.reset(seed=0)
print("info", info)
for tn in ["mujoco_movable_object","mujoco_fixture","mujoco_drawer","mujoco_tidybot_robot"]:
    for t in obs.get_objects(T(tn)):
        print(tn, t.name, {f: round(float(obs.get(t,f)),3) for f in os_.type_features[T(tn)]})
env.close()
