import numpy as np
from env_client import make_env
from probe_fk_ctl import servo, grip_hold, ee_world
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env(); obs,_=env.reset(seed=0); obs=np.asarray(obs,float)
i=3; p=obs[:80].reshape(5,16)[i,:3].copy()
obs,_,_=servo(env,obs,[p[0],p[1],0.60],grip=0.0,steps=90)
obs,_,_=servo(env,obs,[p[0],p[1],float(p[2])],grip=0.0,steps=90)
obs=grip_hold(env,obs,1.0,15)
obs,_,_=servo(env,obs,[p[0],p[1],0.65],grip=1.0,steps=90)
rews=set(); nz=[]
for y in [-0.6,-0.3,0.0,0.3,0.6]:
    obs,e,_=servo(env,obs,[1.02,y,0.60],grip=1.0,steps=50)
    for z in [0.40,0.28,0.16]:
        obs,err,_=servo(env,obs,[1.02,y,z],grip=1.0,steps=45)
        o,r,t,tr,_=env.step(np.concatenate([np.zeros(10),[1.0]])); obs=np.asarray(o,float)
        cc=obs[:80].reshape(5,16)[i,:3]; rews.add(round(float(r),5))
        if abs(r+1)>1e-6: nz.append((round(float(r),4),np.round(cc,3).tolist()))
        print(y,z,"r",round(float(r),4),"cube",np.round(cc,3),"err",round(err,3),"drawer",np.round(obs[103:109],2))
print("rewards seen",sorted(rews),"nonstd",nz)
env.close()
