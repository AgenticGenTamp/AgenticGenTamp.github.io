import numpy as np, sys
from calib_util import *
T=-0.224
DZ=float(sys.argv[4]) if len(sys.argv)>4 else 0.010
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
N=int(sys.argv[2]) if len(sys.argv)>2 else 2
MODE=sys.argv[3] if len(sys.argv)>3 else 'gg'
import calib_util
from env_client import make_env
r=R.__new__(R); r.env=make_env(); r.obs,info0=r.env.reset(seed=seed,options={'object_count':N}); r.qi=r.q().copy(); r.g=0.0; r.rew=[]; QH=r.qi.copy()
print('reset info',info0)
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
    q,e=r.ik_world(np.array(p,float),tool=T)
    if np.max(np.abs(e))>1e-3 if np.ndim(e) else e>1e-3: print('IKERR',p,e)
    r.goto_q(q,settle=settle)
def pick(c):
    phase[0]='pick_'+c
    p=r.P(c); print('pick',c,p.round(4))
    r.gripper(0.0,10)
    r.goto_q(QH,settle=3)
    go([p[0],p[1],0.6])
    for z in np.arange(0.58,p[2]-DZ,-0.01): go([p[0],p[1],z],3)
    go([p[0],p[1],p[2]-DZ],8)
    fk=r.fk(tool=T); print('grasp fk',fk[0].round(4),'cube',r.P(c).round(4),'base',r.base().round(3),'grip',round(r.grip(),3),'qerr',np.abs(r.qi-r.q()).max().round(3))
    r.gripper(1.0,15)
    go([p[0],p[1],p[2]+0.03],5); go([p[0],p[1],0.6],5)
    print('lifted',c,r.P(c).round(4),'grip',round(r.grip(),3))
def place(c,xy,z=0.6,wait=40):
    phase[0]='carry_'+c
    cur=r.fk(tool=T)[0]
    for k in range(1,6): go(cur+(np.array([xy[0],xy[1],z])-cur)*k/5,3)
    print('over',xy,'cube at',r.P(c).round(4))
    phase[0]='release_'+c; r.gripper(0.0,wait)
    print('after release',c,r.P(c).round(4),'others off-floor',[(o,r.P(o).round(3)) for o in cubes if r.P(o)[2]>0.49])
def iso(exclude=()):
    P={c:r.P(c) for c in cubes}
    best=None
    for c in cubes:
        if c in exclude: continue
        p=P[c]
        if abs(p[1]+0.2)>0.13 or abs(p[0]-0.5)>0.2: continue
        d=min([9]+[np.linalg.norm(P[o][:2]-p[:2]) for o in cubes if o!=c])
        if best is None or d>best[1]: best=(c,d)
    return best[0]
phase[0]='drive'; r.goto_base([-0.14,0.0,0.0])
print({c:r.P(c).round(3) for c in cubes})
targets={'g':(0.5,0.2),'0':(0.46,0.17),'1':(0.54,0.17),'2':(0.46,0.23),'3':(0.54,0.23),'4':(0.5,0.26),'5':(0.5,0.14),'6':(0.42,0.2),'7':(0.58,0.2),'8':(0.44,0.26),'c':(0.35,0.1),'i':(0.5,0.0),'e':(0.4,0.14)}
done=[]
for k,m in enumerate(MODE):
    if len(done)>=len(cubes): break
    c=iso(done); pick(c); place(c,targets[m]); done.append(c)
    phase[0]='idle%d'%k
    for t in range(30): r.step(dq=np.zeros(7))
print('green bin',r.P('bin_green_0').round(4))
for c in done: print(c,r.P(c).round(4))
ing=[c for c in cubes if abs(r.P(c)[0]-0.5)<0.225 and abs(r.P(c)[1]-0.2)<0.15]
print('cubes in green footprint',ing,[r.P(c).round(3) for c in ing])
from collections import Counter
print(Counter((ph,rw) for ph,rw,_,_ in LOG))
