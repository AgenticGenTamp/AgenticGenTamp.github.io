import numpy as np, sys
from env_client import make_env
seed=int(sys.argv[1])
env=make_env(); o,_=env.reset(seed=seed)
ds=[]
for t in range(300):
    ds.append(np.linalg.norm(o[16:18]-o[0:2]))
    o,*_=env.step(np.zeros(11,np.float32))
ds=np.array(ds)
print(seed,'bowl',o[0:2].round(3),'d range',ds.min().round(3),ds.max().round(3),'frac<.12',(ds<0.12).mean().round(2), 'd every20',ds[::20].round(2))
