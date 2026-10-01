from env_client import make_env
import numpy as np
env=make_env(); T=env.observation_space.get_type
obs,info=env.reset(seed=1)
try:
    s=env.get_state(); print("get_state ok", type(s))
    try:
        names=s.get_object_names(); print(sorted(names))
    except Exception as e: print("names err",e)
except Exception as e: print("get_state err",repr(e)[:200])
try:
    env.set_state(s); print("set_state ok")
except Exception as e: print("set_state err", repr(e)[:200])
env.close()
