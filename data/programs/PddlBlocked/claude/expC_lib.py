import numpy as np, fk
from env_client import make_env
from lib_util import robot, step_to

BASE = np.array([3.72, 0.10, 0.0])
R0 = fk.grasp_R(0.0)
PLATE = np.array([4.5, -0.3])

def opos(obs,name): return obs.data[obs.get_object_from_name(name)][:3].copy()
def names(obs): return [o.name for o in obs.data]

def gripper(env,obs,v):
    a=np.zeros(11,dtype=np.float32); a[10]=v
    obs,rew,term,trunc,_=env.step(a); return obs,term

def grasp_blocker(env, log):
    obs,_=env.reset(seed=1)
    blk=opos(obs,"blocker")
    obs,rej,n0=step_to(env,obs,BASE,robot(obs)[3:10])
    obs,_=gripper(env,obs,1.0)
    tgt=blk-np.array([0.03,0,0]); pre=tgt-np.array([0.15,0,0])
    q_pre,e1=fk.ik(pre,R0,BASE,robot(obs)[3:10],seeds=6)
    q_g,e2=fk.ik(tgt,R0,BASE,q_pre,seeds=1)
    obs,r1,n1=step_to(env,obs,BASE,q_pre)
    obs,r2,n2=move_tool(env,obs,BASE,tgt,R0,seg=0.03)
    obs,_=gripper(env,obs,-1.0)
    ga=robot(obs)[11]
    log(f"grasp: base_steps={n0} pre={n1}(rej{int(r1)}) grasp={n2}(rej{int(r2)}) ik_err={e1:.4f}/{e2:.4f} ga={ga}")
    return obs,q_g,blk

def tool(obs,base=None):
    r=robot(obs); b=r[:3] if base is None else base
    return fk.world_fk(r[3:10],b)[0]

def move_tool(env,obs,base,tgt_p,R,seg=0.05,grip=0.0,maxseg=40):
    """Cartesian straight-line servo: IK each waypoint from current q (seeds=1)."""
    cur=tool(obs,base); d=np.asarray(tgt_p)-cur; n=max(1,int(np.ceil(np.linalg.norm(d)/seg)))
    n=min(n,maxseg); tot=0; rej=False
    for i in range(1,n+1):
        wp=cur+d*i/n
        q,e=fk.ik(wp,R,base,robot(obs)[3:10],seeds=1)
        obs,r,k=step_to(env,obs,base,q,grip=grip); tot+=k
        if r: rej=True; break
    return obs,rej,tot
