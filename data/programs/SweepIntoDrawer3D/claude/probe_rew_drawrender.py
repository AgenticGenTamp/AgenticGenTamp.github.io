import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0)
for j in range(6):
    o=obs.copy(); o[103+j]=0.45
    print(j, env.render_state(state=o.tolist(), label="drawer%d"%j))
o=obs.copy(); o[103:109]=0.45
print("all", env.render_state(state=o.tolist(), label="drawerall"))
env.close()
