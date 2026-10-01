from env_client import make_env
import numpy as np
env = make_env()
for opts in [{'object_count':5}, {'object_count':1}, {'num_objects':4}]:
    try:
        obs, info = env.reset(seed=0, options=opts)
        n=len([n for n in obs.get_object_names() if n.startswith('cuboid')])
        nc=len([n for n in obs.get_object_names() if n.startswith('cupboard')])
        ys=sorted(round(float(obs.data[obs.get_object_from_name(c)][1]),3) for c in obs.get_object_names() if c.startswith('cupboard'))
        print("OPTS",opts,"-> info",info,"ncub",n,"ncup",nc,"ys",ys)
    except Exception as e:
        print("OPTS",opts,"ERROR",type(e).__name__,str(e)[:300])
env.close()
