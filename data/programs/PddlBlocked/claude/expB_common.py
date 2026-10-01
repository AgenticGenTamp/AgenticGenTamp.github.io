import numpy as np, fk
from lib_util import robot, step_to

def scene(obs):
    d={}
    for o in obs.data:
        d[o.name]=np.array(obs.data[o])
    return d

def names(obs):
    return sorted(o.name for o in obs.data)

def setup(env, seed=1):
    obs,_=env.reset(seed=seed)
    s=scene(obs)
    g=s["green0"][:3]; b=s["blocker"][:3]
    dirv=g[:2]-b[:2]; dirv=dirv/np.linalg.norm(dirv)
    d3=np.array([dirv[0],dirv[1],0.0])
    yaw=float(np.arctan2(dirv[1],dirv[0]))
    return obs,s,g,b,d3,yaw

def grip(env,obs,v):
    a=np.zeros(11,dtype=np.float32); a[10]=v
    obs,_,_,_,_=env.step(a); return obs

def blockpos(obs,name):
    return np.array(obs.data[obs.get_object_from_name(name)][:3])

def move_cart(env, obs, R, target_p, base, grip=0.0, step=0.02, maxseg=60):
    """Move tool along straight cartesian line in small increments, staying in IK branch."""
    import numpy as np, fk
    from lib_util import robot, step_to
    tot=0
    p0,_=fk.world_fk(robot(obs)[3:10], robot(obs)[:3])
    d=target_p-p0; L=np.linalg.norm(d)
    nseg=max(1,int(np.ceil(L/step)))
    q=robot(obs)[3:10].copy()
    for i in range(1,nseg+1):
        wp=p0+d*(i/nseg)
        qn,e=fk.ik(wp,R,base,q,seeds=1,w_rot=1.0,q_ref=q,w_ref=0.02)
        if e>0.01:
            qn,e=fk.ik(wp,R,base,q,seeds=4,w_rot=1.0)
        obs,rej,n=step_to(env,obs,base,qn,grip=grip,maxsteps=15); tot+=n
        p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
        if rej or np.linalg.norm(p-wp)>0.03:
            return obs, True, tot, p
        q=robot(obs)[3:10].copy()
    p,_=fk.world_fk(robot(obs)[3:10],robot(obs)[:3])
    return obs, False, tot, p
