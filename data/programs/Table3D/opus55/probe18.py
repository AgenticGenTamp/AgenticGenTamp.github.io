from env_client import make_env
import numpy as np
env=make_env()
for rot in [0,0.4,-0.4,0.8,1.57]:
    obs,_=env.reset(seed=0); r=obs.get_object_from_name('robot')
    a=np.zeros(11,dtype=np.float32); 
    rr=0
    while abs(rr-rot)>1e-6:
        a=np.zeros(11,dtype=np.float32); a[2]=np.clip(rot-rr,-.4,.4); obs,*_=env.step(a); rr=obs.get(r,'pos_base_rot')
    for k in range(40):
        a=np.zeros(11,dtype=np.float32); a[0]=0.01; obs,*_=env.step(a)
    print(rot, 'max bx', round(obs.get(r,'pos_base_x'),3))
env.close()
