import numpy as np, sys
from env_client import make_env
from kin import *
from concurrent.futures import ThreadPoolExecutor
np.set_printoptions(precision=3, suppress=True, linewidth=150)
def trial(args):
    ci, dyaw, bpos = args
    env=make_env(); obs,_=env.reset(seed=0)
    def goto(qd,bd,n=60,g=0):
        nonlocal obs
        for t in range(n):
            a=np.zeros(11,np.float32); a[3:10]=np.clip(qd-obs[128:135],-.1,.1); a[0:3]=np.clip(bd-obs[125:128],-.1,.1); a[10]=g
            obs,*_=env.step(a)
            if np.max(np.abs(obs[128:135]-qd))<0.005: break
    B=np.array(bpos); c=obs[16*ci:16*ci+3]; yaw=2*np.arctan2(obs[16*ci+6],obs[16*ci+3])+dyaw
    yaw=min([yaw+k*np.pi for k in range(-3,4)], key=lambda a: abs(wrap(a-np.pi)) if False else abs(((a-np.pi)+np.pi)%(2*np.pi)-np.pi))
    q=obs[128:135].copy()
    for z in [0.55,0.50,0.47,0.45]:
        q,_=ik(B,q,np.array([c[0],c[1],z]),R_down(yaw)); goto(q,B)
    zr=fk_world(obs[125:128],obs[128:135])[2,3]
    for _ in range(5): goto(q,B,1,1)
    q2,_=ik(B,q,np.array([c[0],c[1],0.53]),R_down(yaw)); goto(q2,B,40,1)
    env.close()
    return ci, round(dyaw,2), bpos, "zreach %.3f"%zr, "cube z %.3f"%obs[16*ci+2], "yaw %.2f"%yaw
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
jobs=[(ci,dy,bp) for ci in [1,0] for dy in [0,np.pi/2] for bp in [(1.25,-0.7,np.pi),(1.45,0,np.pi)]]
with ThreadPoolExecutor(8) as ex:
    for r in ex.map(trial,jobs): print(r)
