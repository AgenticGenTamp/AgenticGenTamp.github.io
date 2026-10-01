import numpy as np, sys
from env_client import make_env
import approach as A
bn=sys.argv[1]
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(None,None,{}); ap.bin_order=[bn]*4; ap.reset(obs,info)
rs=set()
for t in range(1000):
    a=ap.get_action(obs); obs,r,te,tr,i=env.step(a); rs.add(round(r,4))
    if te: print(bn,'TERMINATED at',t); break
print(bn,'rewards',sorted(rs),'cubes',{n:np.round(A.obj_pos(obs,n),3) for n in obs.get_object_names() if n.startswith('cube')}, flush=True)
env.close()
