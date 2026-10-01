from env_client import make_env
import numpy as np
import approach as A
env=make_env()
for seed in [0,1,2,5,9]:
    for oc in [4,20]:
        obs,info=env.reset(seed=seed, options={'object_count':oc})
        d={n:np.round(A.obj_pos(obs,n),3) for n in sorted(obs.get_object_names()) if n.startswith('bin')}
        print(seed,oc,d)
env.close()
