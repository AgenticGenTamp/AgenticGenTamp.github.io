import numpy as np, fk, sys
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
base=np.array([3.72,0.10,0.0])
def grasp_blocker(env):
    obs,s,g,b,d3,yaw=setup(env,1); R=fk.grasp_R(yaw); tot=0
    obs,_,n=step_to(env,obs,base,robot(obs)[3:10]); tot+=n
    q,_=fk.ik(b-d3*0.20,R,base,robot(obs)[3:10],seeds=6)
    obs,_,n=step_to(env,obs,base,q); tot+=n
    q,_=fk.ik(b-d3*0.03,R,base,q,seeds=6)
    obs,_,n=step_to(env,obs,base,q); tot+=n
    obs=grip(env,obs,-1.0); tot+=1
    return obs,g,b,d3,yaw,R,tot
def cart(env,obs,R,tp,grip=0.0,step=0.02,tag=""):
    tot=0; p0,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    d=tp-p0; L=np.linalg.norm(d); ns=max(1,int(np.ceil(L/step)))
    q=robot(obs)[3:10].copy()
    for i in range(1,ns+1):
        wp=p0+d*(i/ns)
        qn,e=fk.ik(wp,R,base,q,seeds=1,w_rot=1.0,q_ref=q,w_ref=0.02)
        alt=False
        if e>0.008:
            qn2,e2=fk.ik(wp,R,base,q,seeds=6,w_rot=1.0)
            if e2<e: qn,e,alt=qn2,e2,True
        obs,rej,n=step_to(env,obs,base,qn,grip=grip,maxsteps=15); tot+=n
        p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
        dev=np.linalg.norm(p-wp)
        if rej or dev>0.03:
            print(f"   {tag} STOP at seg {i}/{ns} wp={np.round(wp,3).tolist()} rej={rej} ikerr={e:.4f} alt={alt} dev={dev:.3f} tool={np.round(p,3).tolist()}")
            return obs,True,tot,p
        q=robot(obs)[3:10].copy()
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    print(f"   {tag} OK n={tot} tool={np.round(p,3).tolist()}")
    return obs,False,tot,p

env=make_env(); obs,g,b,d3,yaw,R,tot=grasp_blocker(env)
perp=np.array([-d3[1],d3[0],0.0]); p0,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
print("grasped ga",robot(obs)[11],"tot",tot,"perp",np.round(perp,3))
obs,r,n,p=cart(env,obs,R,p0+np.array([0,0,0.05]),tag="lift"); tot+=n
obs,r,n,p=cart(env,obs,R,p+perp*0.30,tag="lat+0.30"); tot+=n
obs,r,n,p=cart(env,obs,R,p-np.array([0,0,0.05]),tag="down"); tot+=n
before=blockpos(obs,'blocker')
obs=grip(env,obs,1.0); tot+=1
print("open1 ga",robot(obs)[11],"opening",round(float(robot(obs)[10]),3),"blk",np.round(blockpos(obs,'blocker'),3).tolist(),"tot",tot)
env.close()
