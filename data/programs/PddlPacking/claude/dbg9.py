import numpy as np
from env_client import make_env
from approach import GeneratedApproach, Robot, rotz
import fk
env=make_env(); obs,info=env.reset(seed=2,options={"object_count":4})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
prev=None
for k in range(90):
    a=ap.get_action(obs)
    R=Robot(obs)
    rej = prev is not None and np.allclose(prev, R.r[:10],atol=1e-12) and np.abs(a[:10]).max()>1e-9
    if 66<=k<=80:
        pw,Rw,frames=R.tool()
        Rb=rotz(R.base[2]); off=np.array([R.base[0],R.base[1],0])
        pts=[np.round(Rb@f[0]+off,3) for f in frames]
        print(k,ap.phase,"REJ" if rej else "   ","tool",np.round(pw,3))
        print("     links",pts[1],pts[3],pts[5],pts[6])
        qn=R.q+np.clip(a[3:10],-0.2,0.2)
        pn,Rn,fn=fk.fk_arm(qn)
        pts2=[np.round(Rb@f[0]+off,3) for f in fn]
        print("     act",np.round(a[3:10],3),"-> tool",np.round(Rb@pn+off,3),"links",pts2[1],pts2[3],pts2[5])
    prev=R.r[:10].copy()
    obs,r,t,tr,i=env.step(a)
env.close()
