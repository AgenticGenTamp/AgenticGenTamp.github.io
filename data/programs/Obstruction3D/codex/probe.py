from env_client import make_env
import numpy as np


def dump(state):
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        vals = {}
        for f in env.observation_space.type_features[obj.type]:
            vals[f] = round(float(state.get(obj, f)), 5)
        print(name, obj.type.name, vals)


env = make_env()
obs, info = env.reset(seed=0)
print("max_steps", env.max_steps, "info", info)
dump(obs)
action = np.zeros(env.action_space.shape, dtype=np.float32)
for i in range(3):
    obs, r, term, trunc, info = env.step(action)
    print("step", i, r, term, trunc)
dump(obs)
env.close()
