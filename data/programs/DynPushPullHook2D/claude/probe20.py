from env_client import make_env
from ctl import act, rget, oget
from grasp import grasp_seq, wrap
import numpy as np
env=make_env()
for seed in [42,0,5,7]:
    obs,info=env.reset(seed=seed)
    obs,n,ok=grasp_seq(env,obs)
    if not ok: print(seed,"grasp FAIL"); continue
    term=[False]
    def step(a):
        global obs
        obs,r,t,tr,i=env.step(a); term[0]=term[0] or t
        return t
    def R(f): return rget(obs,f)
    def B(f): return oget(obs,'target_block',f)
    def H(f): return oget(obs,'hook',f)
    cnt=[n]
    def move(tx,ty,tol=0.004,maxit=400):
        for _ in range(maxit):
            dx=np.clip(tx-R('x'),-0.05,0.05); dy=np.clip(ty-R('y'),-0.05,0.05)
            if abs(dx)<tol and abs(dy)<tol: return
            step(act(dx=dx,dy=dy)); cnt[0]+=1
            if term[0]: return
    def rot(t,tol=0.004):
        for _ in range(300):
            d=wrap(t-R('theta'))
            if abs(d)<tol: return
            step(act(dth=d)); cnt[0]+=1
            if term[0]: return
    # go bottom-left
    move(0.28,0.25)
    # rotate CW 270 to +pi/2
    for tgt in [-1.0,-2.0,-3.0,-4.0,-4.712]:
        rot(tgt)
    print(seed,"after rot robot",round(R('x'),3),round(R('y'),3),round(R('theta'),3),"hook",round(H('x'),3),round(H('y'),3),round(H('theta'),3),"held",H('held'),"block",round(B('x'),3),round(B('y'),3))
    # compute target
    hw=B('width')/2; hh=B('height')/2; th=B('theta')
    ext=abs(hw*np.sin(th))+abs(hh*np.cos(th))
    top=B('y')+ext
    basey=top+0.07-1.95
    basex=B('x')-hw*abs(np.cos(th))-abs(hh*np.sin(th))-0.12
    print("   want base",round(basex,3),round(basey,3),"top",round(top,3))
    move(0.28,np.clip(basey,0.25,1.48))
    move(np.clip(basex,0.28,3.25),np.clip(basey,0.25,1.48))
    print("   at",round(R('x'),3),round(R('y'),3),"block",round(B('x'),3),round(B('y'),3),"steps",cnt[0])
    for _ in range(60):
        if step(act(dy=-0.05)): break
        cnt[0]+=1
    print("   RESULT term",term[0],"steps",cnt[0],"block",round(B('x'),3),round(B('y'),3))
env.close()
