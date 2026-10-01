import numpy as np, sys
from calib_util import *
T=-0.224
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
r=R(seed)
LOG=[]; phase=['init']
def step(db=(0,0,0), dq=None):
    a=np.zeros(11); a[0:3]=np.clip(db,-.1,.1)
    if dq is not None:
        a[3:10]=np.clip(dq,-.1,.1); r.qi=r.qi+0.25*a[3:10]
    a[10]=r.g
    r.obs,rw,t,tr,info=r.env.step(a); r.rew.append(rw)
    if not LOG or LOG[-1][1]!=rw or t or tr:
        print(len(r.rew),phase[0],'rew',rw,'term',t,'trunc',tr,'info',info,flush=True)
    LOG.append((phase[0],rw,t,tr)); return rw
r.step=step
cubes=sorted(n for n in r.obs.get_object_names() if n.startswith('cube'))
print('ncubes',len(cubes))
def go(p,settle=5):
    q,e=r.ik_world(np.array(p,float),tool=T); r.goto_q(q,settle=settle)
def pick(c):
    phase[0]='pick_'+c
    p=r.P(c); print('pick',c,p.round(4))
    r.gripper(0.0,10)
    go([p[0],p[1],0.6])
    for z in np.arange(0.58,p[2]-0.017,-0.01): go([p[0],p[1],z],3)
    go([p[0],p[1],p[2]-0.017],8)
    r.gripper(1.0,15)
    go([p[0],p[1],p[2]+0.03],5); go([p[0],p[1],0.6],5)
    print('lifted',c,r.P(c).round(4),'grip',round(r.grip(),3))
def place(c,xy,z=0.6,wait=40):
    phase[0]='carry_'+c
    cur=r.fk(tool=T)[0]
    for k in range(1,6): go(cur+(np.array([xy[0],xy[1],z])-cur)*k/5,3)
    print('over',xy,'cube at',r.P(c).round(4))
    phase[0]='release_'+c; r.gripper(0.0,wait)
    print('after release',c,r.P(c).round(4))
def iso(exclude=()):
    P={c:r.P(c) for c in cubes}
    best=None
    for c in cubes:
        if c in exclude: continue
        p=P[c]
        if abs(p[1]+0.2)>0.1 or abs(p[0]-0.5)>0.15: continue
        d=min(np.linalg.norm(P[o][:2]-p[:2]) for o in cubes if o!=c)
        if best is None or d>best[1]: best=(c,d)
    return best[0]
phase[0]='drive'; r.goto_base([-0.14,0.0,0.0])
c1=iso(); pick(c1)
place(c1,(0.5,0.2))
phase[0]='idle1'
for k in range(30): r.step(dq=np.zeros(7))
c2=iso([c1]); pick(c2)
place(c2,(0.35,0.1))
phase[0]='idle2'
for k in range(30): r.step(dq=np.zeros(7))
print('green bin',r.P('bin_green_0').round(4))
for c in [c1,c2]: print(c,r.P(c).round(4))
ing=[c for c in cubes if abs(r.P(c)[0]-0.5)<0.225 and abs(r.P(c)[1]-0.2)<0.15]
print('cubes in green footprint',ing)
from collections import Counter
print(Counter((ph,rw) for ph,rw,_,_ in LOG))
