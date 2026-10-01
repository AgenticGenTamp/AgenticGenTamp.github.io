import numpy as np
from env_client import make_env
for n in [1,2,3,5]:
    try:
        env=make_env(); obs,info=env.reset(seed=0,options={"object_count":n})
        o,r,t,tr,i=env.step(np.zeros(11))
        c=obs[:80].reshape(5,16)[:,:3]
        print("n",n,"r",r,"cubes",np.round(c,3).tolist(),"info",info,i)
        env.close()
    except Exception as e:
        print("n",n,"ERR",type(e).__name__,str(e)[:200])
