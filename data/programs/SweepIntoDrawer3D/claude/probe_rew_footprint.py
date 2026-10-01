import numpy as np
from env_client import make_env
res=[]
for ytar in [-1.2,-0.9,-0.6,-0.3,0.0,0.3,0.6,0.9,1.2]:
    env=make_env(); obs,_=env.reset(seed=0)
    for k in range(40):
        a=np.zeros(11); a[1]=np.clip(ytar-obs[126],-0.1,0.1); a[0]=np.clip(1.6-obs[125],-0.1,0.1)
        obs,r,t,tr,_=env.step(a)
    for k in range(40):
        a=np.zeros(11); a[0]=-0.1; a[1]=np.clip(ytar-obs[126],-0.1,0.1)
        obs,r,t,tr,_=env.step(a)
    res.append((ytar,round(float(obs[125]),3),round(float(obs[126]),3),r))
    env.close()
for r in res: print("y_target",r[0],"blocked_x",r[1],"y",r[2],"rew",r[3])
