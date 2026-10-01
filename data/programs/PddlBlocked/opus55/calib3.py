from env_client import make_env
import numpy as np, sys
from kin import *
from envutil import Sim
env = make_env()
S=Sim(env,env.reset(seed=0)[0])
blk=S.block('blocker'); g0=S.block('green0')
d=g0[:2]-blk[:2]; d=np.r_[d/np.linalg.norm(d),0]
base=np.array([3.85, blk[1]-0.2, 0.0])
print('reach base', S.moveto(base,S.q()), S.base())
for h in [0,-0.04,0.04]:
    tgt=blk[:3]+np.array([0,0,h])
    pre=tgt-0.15*d
    b,q,err=ik(pre+[0,0,0.15],d,base,S.q(),free_base=False,elbow_min=0.9)
    print('h',h,'err',round(err,4),'up ok',S.moveto(b,q,grip=1.0))
    b,q,err=ik(pre,d,base,q,free_base=False,elbow_min=0.9)
    print(' pre ok',S.moveto(b,q))
    b2,q2,err2=ik(tgt,d,b,q,free_base=False,elbow_min=0.9)
    ok=S.moveto(b2,q2,grip=0,maxd=0.05); print(' in reached',ok,'err',round(err2,4))
    S.step(np.r_[np.zeros(10),-1.0])
    ga=S.rget('grasp_active'); print(' grasp',ga,[round(S.rget(f),4) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']])
    if ga>0.5:
        p,R=fk_world(S.base(),S.q()); print(' fk tool',p.round(4),'block',S.block('blocker')[:3].round(4)); print(R.round(3))
        break
    S.step(np.r_[np.zeros(10),1.0])
    S.moveto(b,q,grip=0)
print('steps',S.steps)
# lift
b,q,_=ik(S.block('blocker')[:3]+[0,0,0.12]-0.0*d,d,S.base(),S.q(),free_base=False,elbow_min=0.9)
print('lift',S.moveto(b,q,maxd=0.05))
def show(tag):
    p,R=fk_world(S.base(),S.q()); bl=S.block('blocker')
    print(tag,'fk tool(t=.17)',p.round(4),'Rz',R[:,2].round(3),'block',bl[:3].round(4),'diff',(bl[:3]-p).round(4))
show('lifted')
for k in range(16):
    a=np.zeros(11); a[9]=0.2; S.step(a)
show('roll+3.2')
for k in range(5):
    a=np.zeros(11); a[3]=-0.1; a[4]=-0.1; a[8]=-0.1; S.step(a)
show('moved')
