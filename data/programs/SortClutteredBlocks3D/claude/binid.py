import numpy as np, sys
from env_client import make_env
import approach as A
b_name=sys.argv[1]; seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
ap=A.GeneratedApproach(None,None,{}); ap.bin_order=[b_name]*4
ap.reset(obs,info)
rews=[]
for t in range(260):
    a=ap.get_action(obs); obs,r,te,tr,i=env.step(a); rews.append(round(r,3))
    if ap.idx>0: break
print(b_name, 'cube1 end',np.round(A.obj_pos(obs,'cube1'),3), 'last rews',rews[-6:], 'minrew',min(rews),'maxrew',max(rews), flush=True)
env.close()
