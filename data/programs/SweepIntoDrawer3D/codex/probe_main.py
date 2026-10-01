import numpy as np
from env_client import make_env

for seed in range(3):
    e=make_env(); o,info=e.reset(seed=seed)
    print('SEED',seed,'info',info,'max',e.max_steps)
    print('cubes',np.array([[o[16*i],o[16*i+1],o[16*i+2],*o[16*i+13:16*i+16]] for i in range(5)]))
    print('cook',o[80:89],'island',o[96:109],'robot',o[125:136],'wiper',o[147:163])
    a=np.zeros(11,np.float32); oo,r,t,tr,inf=e.step(a)
    print('zero reward',r,t,tr,inf,'delta robot',oo[125:136]-o[125:136],'delta wiper',oo[147:163]-o[147:163])
    e.close()
