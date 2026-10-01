import numpy as np
from env_client import make_env

def chairs(obs):
    names=[o.name for o in obs]
    return sorted([n for n in names if 'chair' in n])

def cxy(obs,name):
    o=obs.get_object_from_name(name); return np.array([obs.get(o,'x'),obs.get(o,'y')])

def cfull(obs,name):
    o=obs.get_object_from_name(name)
    return {f:float(obs.get(o,f)) for f in ['x','y','z','qw','qx','qy','qz','vx','vy','vz','wx','wy','wz','bb_x','bb_y','bb_z']}

def rxy(obs):
    r=obs.get_object_from_name('robot'); return np.array([obs.get(r,'pos_base_x'),obs.get(r,'pos_base_y')])

def act(dx=0.,dy=0.,drot=0.):
    a=np.zeros(11,dtype=np.float32); a[0]=dx;a[1]=dy;a[2]=drot; return a

def move_to(env,obs,target,maxsteps=200,step=0.1,tol=0.03,log=None):
    """move base straight toward target; returns (obs,term,steps)"""
    for i in range(maxsteps):
        p=rxy(obs); d=target-p; n=np.linalg.norm(d)
        if n<tol: return obs,False,i
        v=d/n*min(step,n)
        obs,r,term,trunc,info=env.step(act(v[0],v[1]))
        if log is not None: log.append(obs)
        if term or trunc: return obs,term,i+1
    return obs,False,maxsteps

def orbit_to(env,obs,center,ang_target,R,dstep=0.06,log=None):
    """move around center at radius R to angle ang_target"""
    p=rxy(obs); a0=np.arctan2(*(p-center)[::-1])
    da=(ang_target-a0+np.pi)%(2*np.pi)-np.pi
    n=max(1,int(abs(da)*R/dstep))
    for k in range(1,n+1):
        a=a0+da*k/n
        tgt=center+R*np.array([np.cos(a),np.sin(a)])
        obs,term,_=move_to(env,obs,tgt,maxsteps=6,tol=0.02,log=log)
        if term: return obs,True
    return obs,False
