from env_client import make_env
import numpy as np
env = make_env()
for n in [1,2,3,4,5,6,8]:
    try:
        obs, info = env.reset(seed=3, options={'object_count':n})
        names=obs.get_object_names()
        cups=[c for c in names if c.startswith('cupboard')]
        ys=sorted(round(float(obs.data[obs.get_object_from_name(c)][1]),3) for c in cups)
        cubs=[c for c in names if c.startswith('cuboid')]
        _,r,_,_,_=env.step(np.zeros(11,dtype=np.float32))
        print(f"n={n} ncub={len(cubs)} ncup={len(cups)} ys={ys} r0={r}")
    except Exception as e:
        print(n,"ERR",type(e).__name__,str(e)[:200])
env.close()
