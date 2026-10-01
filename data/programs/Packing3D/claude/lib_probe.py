import numpy as np
from ik import ik
from fk import fk
JNAMES=[f'joint_{i}' for i in range(1,8)]
Rdown = np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
def getq(obs):
    r=obs.get_object_from_name('robot')
    return np.array([obs.get(r,n) for n in JNAMES])
def move_to(env,obs,qd,maxsteps=80,close=0.0):
    blocked=0
    for k in range(maxsteps):
        q=getq(obs); d=qd-q
        if np.max(np.abs(d))<1e-4: break
        a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2); a[10]=close
        prev=q
        obs,rew,term,trunc,info=env.step(a)
        if np.max(np.abs(getq(obs)-prev))<1e-9: blocked+=1
        if blocked>2: break
    return obs, blocked>2
def goto_xyz(env,obs,xyz,R=Rdown,base=(0,0),maxsteps=80):
    q=getq(obs)
    tgt=np.array([xyz[0]-base[0], xyz[1]-base[1], xyz[2]])
    qd,M=ik(q,tgt,R)
    return move_to(env,obs,qd)
