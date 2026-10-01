import numpy as np
from env_client import make_env

def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi

def action_pose(s, pos=None, theta=None, arm=None, vac=0.0):
    a=np.zeros(5,np.float32); a[4]=vac
    if pos is not None:
        d=np.asarray(pos)-s[:2]; a[:2]=np.clip(d,-.05,.05)
    if theta is not None: a[2]=np.clip(wrap(theta-s[2]),-.196,.196)
    if arm is not None: a[3]=np.clip(arm-s[4],-.1,.1)
    return a

def run(seed=0):
    e=make_env(); s,_=e.reset(seed=seed)
    u=np.array([np.cos(s[11]),np.sin(s[11])]); tip=s[9:11]-s[18]*u
    # Approach the long-leg endpoint with gripper pointing along +long-axis.
    heading=np.arctan2(u[1],u[0])
    pre=tip-0.24*u
    phases=[('retract',20),('move',80),('turn',40),('extend',10),('translate',30)]
    print('seed',seed,'tip',tip,'pre',pre,'head',heading,'init rob',s[:5])
    t=0
    for name,n in phases:
        for j in range(n):
            old=s.copy()
            if name=='retract': a=action_pose(s,arm=0)
            elif name=='move': a=action_pose(s,pos=pre,arm=0)
            elif name=='turn': a=action_pose(s,pos=pre,theta=heading,arm=0)
            elif name=='extend': a=action_pose(s,pos=pre,theta=heading,arm=.2,vac=1)
            else: a=action_pose(s,pos=pre+np.array([.3,0]),theta=heading,arm=.2,vac=1)
            s,r,term,trunc,info=e.step(a); t+=1
            moved=np.linalg.norm(s[9:11]-old[9:11])
            if j==0 or j==n-1 or moved>.001:
                print(t,name,j,'rob',np.round(s[[0,1,2,4,6]],3),'hook',np.round(s[[9,10,11]],3),'dm',round(moved,3),'but',np.round(s[20:22],3))
            if term or trunc: break
    e.close()

if __name__=='__main__':
 import sys;run(int(sys.argv[1]) if len(sys.argv)>1 else 0)
