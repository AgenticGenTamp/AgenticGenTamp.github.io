from env_client import make_env
import numpy as np, time
env = make_env()
a=np.zeros(11,dtype=np.float32)
for k in range(3):
    t0=time.time(); obs,info=env.reset(seed=k, options={'object_count':100}); tr=time.time()-t0
    t0=time.time()
    for i in range(20): obs,_,_,_,_=env.step(a)
    print('reset',round(tr,2),'step',round((time.time()-t0)/20,3))
env.close()
