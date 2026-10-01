import numpy as np
from env_client import make_env
env = make_env()
obs, info = env.reset(seed=0)
# do nothing
for i in range(5):
    o,rew,t,tr,inf=env.step(np.zeros(11,dtype=np.float32))
    print(i, rew, t, tr, inf)
env.close()
for seed in [1,2,3,4,5]:
    e=make_env(); ob,inf=e.reset(seed=seed)
    cs=[n for n in ob.get_object_names() if n.startswith('cupboard')]
    cbs=[n for n in ob.get_object_names() if n.startswith('cuboid')]
    print(seed, inf, len(cs), len(cbs))
    for n in sorted(cs):
        oo=ob.get_object_from_name(n); print("  ",n, np.round(ob.data[oo],3))
    for n in sorted(cbs):
        oo=ob.get_object_from_name(n); print("  ",n, np.round(ob.data[oo],3))
    o,rew,t,tr,i2=e.step(np.zeros(11,dtype=np.float32))
    print("   rew0", rew)
    e.close()
