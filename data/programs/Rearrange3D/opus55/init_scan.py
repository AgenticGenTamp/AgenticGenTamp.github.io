import numpy as np, sys
from env_client import make_env
env=make_env(); bad=[]
for s in range(int(sys.argv[1]),int(sys.argv[2])):
    o,_=env.reset(seed=s)
    zd,zc=o[18],o[34]
    flag=[]
    if zd<0.5: flag.append('drink_low%.3f'%zd)
    if zc<0.505: flag.append('can_low%.3f'%zc)
    dd=np.linalg.norm(o[16:18]-o[0:2]); dc=np.linalg.norm(o[32:34]-o[0:2])
    if flag: bad.append((s,flag,round(dd,3),round(dc,3), o[16:18].round(2).tolist()))
print(len(bad)); [print(b) for b in bad]
