import numpy as np, fk
from env_client import make_env
from lib_util import robot, step_to

BASE = np.array([3.72, 0.10, 0.0])
R = fk.grasp_R(0.0)

def new_rig():
    env = make_env()
    obs,_ = env.reset(seed=1)
    blk = obs.data[obs.get_object_from_name("blocker")][:3].copy()
    obs,rej,n = step_to(env, obs, BASE, robot(obs)[3:10])
    return env, obs, blk

def trial(env, obs, blk, a, b, c, nsub=6, pre=0.15):
    tgt = np.array([blk[0]-a, blk[1]+b, blk[2]+c])
    rec = dict(a=round(a,3),b=round(b,3),c=round(c,3))
    q = robot(obs)[3:10]
    p0 = tgt - np.array([pre,0,0])
    qp,e = fk.ik(p0, R, BASE, q, seeds=8)
    rec['ik_pre']=round(float(e),4)
    if e>0.01: rec['status']='ik_fail'; return obs,rec
    z=np.zeros(11,dtype=np.float32); z[10]=1.0
    obs,_,_,_,_ = env.step(z)
    obs,rj,n = step_to(env,obs,BASE,qp)
    rec['rej_pre']=int(rj)
    if rj: rec['status']='rej_pre'; return obs,rec
    # cartesian approach
    qc = qp.copy(); worst=0.0; rejstep=None
    for i in range(1,nsub+1):
        p = p0 + np.array([pre*i/nsub,0,0])
        qi,ei = fk.ik(p, R, BASE, qc, seeds=2)
        worst=max(worst,float(ei))
        if ei>0.01: rec['ik']=round(worst,4); rec['status']='ik_fail'; return obs,rec
        obs,rj,n = step_to(env,obs,BASE,qi)
        if rj: rejstep=i; break
        qc=qi
    rec['ik']=round(worst,4)
    rec['rej_sub']=rejstep
    qa = robot(obs)[3:10]
    rec['reach_err']=round(float(np.linalg.norm(fk.pose_err(qa,BASE,tgt,R)[:3])),4)
    z=np.zeros(11,dtype=np.float32); z[10]=-1.0
    obs,_,_,_,_ = env.step(z)
    rec['ga']=int(robot(obs)[11]>0.5)
    rec['status']='ok' if rejstep is None else 'rej_approach'
    return obs, rec

def reset(env):
    obs,_ = env.reset(seed=1)
    blk = obs.data[obs.get_object_from_name("blocker")][:3].copy()
    obs,rej,n = step_to(env, obs, BASE, robot(obs)[3:10])
    return obs, blk
