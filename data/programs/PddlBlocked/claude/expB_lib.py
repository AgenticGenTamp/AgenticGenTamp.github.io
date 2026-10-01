import numpy as np, fk
from lib_util import robot, step_to
from expB_common import *
BASE=np.array([3.72,0.10,0.0])
def cart(env,obs,R,tp,grip=0.0,step=0.02,tag="",base=BASE,verbose=True):
    tot=0; p0,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    d=tp-p0; L=np.linalg.norm(d); ns=max(1,int(np.ceil(L/step)))
    q=robot(obs)[3:10].copy()
    for i in range(1,ns+1):
        wp=p0+d*(i/ns)
        qn,e=fk.ik(wp,R,base,q,seeds=1,w_rot=1.0,q_ref=q,w_ref=0.02)
        if e>0.008:
            qn2,e2=fk.ik(wp,R,base,q,seeds=6,w_rot=1.0)
            if e2<e: qn,e=qn2,e2
        obs,rej,n=step_to(env,obs,base,qn,grip=grip,maxsteps=15); tot+=n
        p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
        if rej or np.linalg.norm(p-wp)>0.03:
            if verbose: print(f"   {tag} STOP seg{i}/{ns} rej={rej} tool={np.round(p,3).tolist()} n={tot}")
            return obs,True,tot,p
        q=robot(obs)[3:10].copy()
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    if verbose: print(f"   {tag} OK n={tot} tool={np.round(p,3).tolist()}")
    return obs,False,tot,p
def grasp_at(env,obs,R,target,base=BASE,pre=0.20,tag=""):
    tot=0
    q,_=fk.ik(target-R[:,0]*pre,R,base,robot(obs)[3:10],seeds=6)
    obs,r1,n=step_to(env,obs,base,q); tot+=n
    q2,_=fk.ik(target,R,base,q,seeds=6)
    obs,r2,n=step_to(env,obs,base,q2); tot+=n
    obs=grip(env,obs,-1.0); tot+=1
    return obs,tot,(r1,r2)
