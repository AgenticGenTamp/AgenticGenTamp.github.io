import numpy as np
from env_client import make_env
for mode in ('baseline','small','rotate_pos','rotate_neg','joints_pos','joints_neg','diagonal'):
    env=make_env();s,_=env.reset(seed=56)
    print('START',mode,s[:3].tolist(),s[19:21].tolist(),flush=True)
    for k in range(30):
        a=np.zeros(11,dtype=np.float32); limit=.05 if mode=='small' else .4
        a[:2]=np.clip(s[19:21]-s[:2],-limit,limit)
        if mode=='rotate_pos':a[2]=.4
        if mode=='rotate_neg':a[2]=-.4
        if mode=='joints_pos':a[3:10]=.4
        if mode=='joints_neg':a[3:10]=-.4
        if mode=='diagonal' and k<5:a[:2]=(-.4,-.4)
        n,r,t,tr,i=env.step(a)
        if k<6 or k>=25 or t:print('STEP',mode,k,n[:3].tolist(),'d',(n[:3]-s[:3]).tolist(),'t',t,flush=True)
        s=n
        if t or tr:break
    env.close()
