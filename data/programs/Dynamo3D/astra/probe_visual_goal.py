import numpy as np
from env_client import make_env
for count in [3,12,1]:
    e=make_env();s,info=e.reset(seed=0,options={'object_count':count});r=s.get_object_from_name('robot');total=0
    for target in ([(-1,4.7),(3.7,4.7),(3.7,3.7),(3.75,3.75)] if count>1 else [(-.5,1),(1,1),(1,0)]):
        for k in range(150):
            p=np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y')]);d=np.array(target)-p
            if np.linalg.norm(d)<.025:break
            a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(d,-.1,.1)
            s,re,t,tr,inf=e.step(a);total+=1
            if t or tr or re!=-1: print('SPECIAL',count,target,total,p,re,t,tr,flush=True)
            if t or tr:break
        print('TARGET',count,target,total,[s.get(r,'pos_base_x'),s.get(r,'pos_base_y')],re,t,tr,flush=True)
        if t or tr:break
    e.close()
