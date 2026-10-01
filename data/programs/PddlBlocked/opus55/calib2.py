from env_client import make_env
import numpy as np, sys
from kin import *
from envutil import Sim
env = make_env()
obs,_=env.reset(seed=0)
S=Sim(env,obs)
blk=S.block('blocker'); g0=S.block('green0')
d=g0[:2]-blk[:2]; d=np.r_[d/np.linalg.norm(d),0]
bnom=np.array([blk[0]-0.75*d[0]-0.0, blk[1]-0.75*d[1]-0.2, np.arctan2(d[1],d[0])])
for h in [0,-0.05,-0.1,-0.15,-0.2,-0.25,-0.3,0.05]:
    tgt=blk[:3]+np.array([0,0,h])
    pre=tgt-0.15*d
    b,q,err=ik(pre,d,S.base(),S.q(),base_nom=bnom)
    print('h',h,'pre ik',b.round(3),q.round(2),'err',round(err,4))
    ok=S.moveto(b,q,grip=1.0); print(' pre reached',ok, S.base().round(3))
    b2,q2,err2=ik(tgt,d,b,q,free_base=False)
    ok=S.moveto(b2,q2,grip=0,maxd=0.05); print(' in reached',ok,'err',round(err2,4))
    S.step(np.r_[np.zeros(10),-1.0])
    ga=S.rget('grasp_active'); print(' grasp',ga,[round(S.rget(f),4) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']])
    if ga>0.5:
        p,R=fk_world(S.base(),S.q()); print(' fk tool',p.round(4),'block',S.block('blocker')[:3].round(4))
        break
    S.step(np.r_[np.zeros(10),1.0])
    S.moveto(b,q,grip=0)
print('steps',S.steps)
