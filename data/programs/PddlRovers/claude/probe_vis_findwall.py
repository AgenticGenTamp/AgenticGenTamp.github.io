import numpy as np
from probe_vis_lib import *
env=make_env()
for seed in range(60):
    obs,info=env.reset(seed=seed, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    if -1.0<O[0]<-0.30:
        # can rover0 stand east within 2.0 with ray crossing x=0?
        print(seed, np.round(O,3))
env.close()
