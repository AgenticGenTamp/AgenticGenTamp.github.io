"""Carry the verified grasp around to the cupboard's +x opening."""
import numpy as np
from env_client import make_env
from probe_alt_branches import drive, qpos, cube_pos, get
from solve_alt_fk import fk

LOW=np.array([0,2.24,2.945,-1,-.982,.20,1.57])
LIFT=LOW.copy();LIFT[1]=1.00

def drive_pose(env,state,qtarget,bt,grip,steps=100):
    for _ in range(steps):
        a=np.zeros(11,np.float32);qerr=qtarget-qpos(state)
        a[3:10]=np.clip(.35*qerr,-.1,.1)
        b=np.array([get(state,"robot",f) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
        err=np.asarray(bt)-b;err[2]=(err[2]+np.pi)%(2*np.pi)-np.pi
        a[:3]=np.clip(err/.87,-.1,.1);a[10]=grip
        state,reward,term,trunc,_=env.step(a)
        if np.max(np.abs(err))<.02 and np.max(np.abs(qerr))<.03: break
    return state

env=make_env();s,_=env.reset(seed=0,options={"object_count":1});c="cube1"
c0=cube_pos(s,c);b0=np.array([get(s,"robot",f) for f in ("pos_base_x","pos_base_y")])
s=drive(env,s,base_target=b0+[-.15,0],steps=40)
s=drive(env,s,q_target=LOW,base_target=b0+[-.15,0])
p=fk(qpos(s))[:3,3];bt=c0[:2]-np.array([-p[0],p[1]])
s=drive(env,s,q_target=LOW,base_target=bt)
s=drive(env,s,q_target=LOW,base_target=bt,grip=.6,steps=18)
s=drive(env,s,q_target=LIFT,base_target=bt,grip=.6,steps=100)
print("lift",np.round(cube_pos(s,c),3),"base",np.round(bt,3))
for target in ([.05,-.07,0],[.40,-.07,0]):
    s=drive_pose(env,s,LIFT,target,.6,steps=90)
    print("way",target,np.round(cube_pos(s,c),3))
for i in range(40):
    a=np.zeros(11,np.float32);a[10]=0
    s,reward,term,trunc,_=env.step(a)
    if i%5==0:print("rel",i,reward,np.round(cube_pos(s,c),3),term)
    if term or trunc:break
env.close()
