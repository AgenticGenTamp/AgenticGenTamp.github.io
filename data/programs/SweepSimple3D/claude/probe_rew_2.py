from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=0)
for name,fn in [("get_state",lambda: env.get_state()),("make_primitives",lambda: env.make_primitives()),("max_steps",lambda: env.max_steps)]:
    try:
        v=fn(); print(name,"OK", type(v), (np.shape(v) if hasattr(v,'shape') else (list(v)[:20] if hasattr(v,'__iter__') else v)))
    except Exception as e: print(name,"FAIL",repr(e)[:200])
env.close()
