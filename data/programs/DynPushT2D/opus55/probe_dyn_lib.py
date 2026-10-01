from env_client import make_env
import numpy as np
np.set_printoptions(precision=4, suppress=True)
A=0.0499
def rot(t): c,s=np.cos(t),np.sin(t); return np.array([[c,-s],[s,c]])
def w2l(obs,p): return rot(obs[2]).T@(np.asarray(p)-obs[0:2])
def l2w(obs,p): return obs[0:2]+rot(obs[2])@np.asarray(p)
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
class Sim:
    def __init__(s): s.env=make_env()
    def reset(s,seed): s.obs,_=s.env.reset(seed=seed); return s.obs
    def step(s,a):
        a=np.clip(np.asarray(a,float),-A,A)
        s.obs,*_=s.env.step(a); return s.obs
    def moveto(s,tgt,n=400,tol=1e-4,spd=A):
        for i in range(n):
            d=tgt-s.obs[16:18]
            if np.linalg.norm(d)<tol: break
            m=np.max(np.abs(d)); 
            a=d if m<=spd else d/m*spd
            s.step(a)
    def goto_local(s,pl,R=1.05,center=(0,-0.45)):
        """navigate around block via circle of radius R around local center, then to local point pl"""
        o=s.obs; c=np.array(center); b0=o[0:3].copy()
        rl=w2l(o,o[16:18])-c; tl=np.asarray(pl)-c
        a0=np.arctan2(rl[1],rl[0]); a1=np.arctan2(tl[1],tl[0])
        # go radially out to R
        s.moveto(l2w(o,c+R*np.array([np.cos(a0),np.sin(a0)])*max(1,np.linalg.norm(rl)/R)))
        da=wrap(a1-a0); n=int(abs(da)/0.1)+1
        for k in range(1,n+1):
            a=a0+da*k/n; s.moveto(l2w(o,c+R*np.array([np.cos(a),np.sin(a)])))
        s.moveto(l2w(o,pl))
        moved=np.abs(s.obs[0:3]-b0).max()
        return moved
    def push(s,dir_local,speed,n,log=True):
        rows=[]
        for i in range(n):
            o=s.obs.copy(); d=rot(o[2])@np.asarray(dir_local,float); d=d/np.linalg.norm(d)*speed
            o2=s.step(d)
            dl=rot(o[2]).T@(o2[0:2]-o[0:2]); dth=wrap(o2[2]-o[2])
            rp=w2l(o,o[16:18]); rp2=w2l(o2,o2[16:18])
            rows.append(np.r_[dl,dth,rp,rp2,o2[16:18]-o[16:18]])
        return np.array(rows)
