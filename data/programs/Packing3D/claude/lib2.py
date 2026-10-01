import numpy as np
from ik2 import ik_solve
from fk import fk
JNAMES=[f'joint_{i}' for i in range(1,8)]
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
def getq(obs):
    r=obs.get_object_from_name('robot')
    return np.array([obs.get(r,n) for n in JNAMES])
def step_to(env,obs,qd,close=0.0,maxsteps=60):
    blocked=0
    for k in range(maxsteps):
        q=getq(obs); d=qd-q
        if np.max(np.abs(d))<1e-4: break
        a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2); a[10]=close
        obs,rew,term,trunc,info=env.step(a)
        if np.max(np.abs(getq(obs)-q))<1e-9:
            blocked+=1
            if blocked>=2: return obs,True
        else: blocked=0
    return obs,False
def goto(env,obs,xyz,R=Rdown,base=(0.0,0.0),rng=None,yaw=None):
    q=getq(obs)
    RR=R
    if yaw is not None:
        c,s=np.cos(yaw),np.sin(yaw)
        RR=np.array([[c,-s,0],[s,c,0],[0,0,1]])@R
    tgt=np.array([xyz[0]-base[0],xyz[1]-base[1],xyz[2]])
    qd,M,pe,re=ik_solve(q,tgt,RR,rng=rng)
    if pe>3e-3 or re>3e-2:
        return obs,True,(pe,re)
    obs,blocked=step_to(env,obs,qd)
    return obs,blocked,(pe,re)
