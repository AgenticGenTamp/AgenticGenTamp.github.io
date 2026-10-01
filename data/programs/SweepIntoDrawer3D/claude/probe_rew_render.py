import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0)
paths=[]
def snap(o,l): paths.append(env.render_state(state=np.asarray(o,dtype=float).tolist(),label=l))
snap(obs,"t0")
for k in range(15):
    a=np.zeros(11); a[0]=-0.1; obs,r,t,tr,_=env.step(a)
snap(obs,"t1_fwd")
lo=np.array([-2.9,-1.76,-2.9,-3.07,-2.9,-0.02,-2.9]); hi=np.array([2.9,1.76,2.9,-0.07,2.9,3.75,2.9])
targets=[np.array([0,0.6,3.14,-1.6,0,1.2,1.57]),np.array([0,1.0,3.14,-2.0,0,2.5,1.57]),np.array([-1.0,0.6,3.14,-1.6,0,1.2,1.57])]
for j,qt in enumerate(targets):
    for k in range(30):
        a=np.zeros(11); a[3:10]=np.clip(qt-obs[128:135],-0.1,0.1); a[10]=0.0
        obs,r,t,tr,_=env.step(a)
    snap(obs,"t%d_q"%(j+2))
    print(j,"q",np.round(obs[128:135],2),"r",r,"cubes",np.round(obs[:80].reshape(5,16)[:,:3],3).tolist())
print("\n".join(paths))
env.close()
