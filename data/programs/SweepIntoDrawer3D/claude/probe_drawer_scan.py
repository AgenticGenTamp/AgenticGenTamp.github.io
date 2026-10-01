import numpy as np, threading, sys
from env_client import make_env
from fk import fk
from ctrl2 import moveto, move_base, RFWD, OX, L, AZ

def tip_now(obs, R):
    M=fk(obs[128:135])
    p=M[:3,3]+M[:3,:3]@np.array([0,0,L])+np.array([OX,0,0])
    return np.array([p[0],p[1],p[2]+AZ])

BX=1.60
def w2l(obs,wx,wy):
    bx,by,yaw=obs[125:128]
    dx,dy=wx-bx,wy-by
    return np.cos(yaw)*dx+np.sin(yaw)*dy, -np.sin(yaw)*dx+np.cos(yaw)*dy

out={}
lock=threading.Lock()
def job(tag, z, ys):
    env=make_env(); obs,_=env.reset(seed=0)
    obs=move_base(env,obs,[BX,obs[126],obs[127]],steps=45,grip=0.0)
    bx,by,yaw=obs[125:128]
    rows=[]
    for wy in ys:
        lx0,ly=w2l(obs,1.05,wy)
        obs,d0=moveto(env,obs,[lx0,ly,z],R=RFWD,steps=90,grip=0.0)
        lxp,_=w2l(obs,0.70,wy)
        obs,d=moveto(env,obs,[lxp,ly,z],R=RFWD,steps=130,grip=0.0,stall_n=25)
        t=tip_now(obs,RFWD)
        # tip local -> world x
        wx = bx - np.cos(yaw)*t[0] + np.sin(yaw)*t[1]
        wyy = by - np.sin(yaw)*t[0] - np.cos(yaw)*t[1]
        rows.append(dict(wy=round(wy,3),z=round(z,3),hit_wx=round(float(wx),4),
                         tipwy=round(float(wyy),3),tipz=round(float(t[2]),3),
                         err=round(d['err'],3),ikres=round(d['ikres'],4),clip=round(d['clip'],3),
                         dr=list(np.round(obs[103:109],3))))
        # retract
        lxr,_=w2l(obs,1.05,wy)
        obs,_=moveto(env,obs,[lxr,ly,z],R=RFWD,steps=70,grip=0.0)
    env.close()
    with lock: out[tag]=rows

jobs=[]
zs=[0.20,0.26,0.32,0.38,0.44]
for z in zs:
    for gi,ys in enumerate([[-0.45,-0.30,-0.15],[0.0,0.15,0.30]]):
        jobs.append((f"z{z}_g{gi}",z,ys))
ths=[threading.Thread(target=job,args=j) for j in jobs]
[t.start() for t in ths]; [t.join() for t in ths]
for k in sorted(out):
    for r in out[k]: print(k, r)
