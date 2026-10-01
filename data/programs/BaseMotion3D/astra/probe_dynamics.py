import numpy as np
from env_client import make_env

for seed in (0,1,2,10,42):
    env=make_env()
    s,info=env.reset(seed=seed)
    print('RESET',seed,'base',s[:3].tolist(),'target',s[19:22].tolist(),'info',info,flush=True)
    for j,act in enumerate(([.4,0,0],[0,.4,0],[0,0,.4],[.4,0,0],[.4,.4,.4],[-.4,-.4,-.4])):
        a=np.zeros(11,dtype=np.float32);a[:3]=act
        n,r,t,tr,i=env.step(a)
        print('STEP',seed,j,'action',act,'delta',(n[:3]-s[:3]).tolist(),'rtt',r,t,tr,'info',i,flush=True)
        s=n
        if t or tr:break
    env.close()
