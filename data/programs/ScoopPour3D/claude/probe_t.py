from env_client import make_env
import numpy as np, time, sys
oc=int(sys.argv[1])
env = make_env()
t0=time.time(); obs, info = env.reset(seed=1, options={'object_count':oc}); tr=time.time()-t0
a=np.zeros(11,dtype=np.float32)
t0=time.time()
for i in range(20): obs,_,_,_,_=env.step(a)
print('oc',oc,'reset',round(tr,2),'per-step',round((time.time()-t0)/20,3))
env.close()
