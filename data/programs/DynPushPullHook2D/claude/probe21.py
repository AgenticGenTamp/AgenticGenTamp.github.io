from env_client import make_env
from ctl import act, rget, oget
from grasp import grasp_seq, wrap
import numpy as np
env=make_env()
for seed in [3,4,6,1]:
    obs,info=env.reset(seed=seed)
    def step(a):
        global obs
        obs,_,_,_,_=env.step(a)
    def R(f): return rget(obs,f)
    def H(f): return oget(obs,'hook',f)
    n=0
    def fe():
        hth=H('theta'); u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
        return np.array([H('x'),H('y')])-1.4*u-0.0533*nv
    print(seed,"hook0",round(H('x'),3),round(H('y'),3),round(H('theta'),3),"fe",np.round(fe(),3))
    # orient arm away
    for _ in range(80):
        d=wrap(H('theta')+np.pi-R('theta'))
        if abs(d)<0.01: break
        step(act(dth=d,da=-0.1,dg=-0.02)); n+=1
    f=fe(); 
    # travel to left of fe
    for _ in range(300):
        dx=np.clip(f[0]-0.45-R('x'),-0.05,0.05); dy=np.clip(0.26-R('y'),-0.05,0.05)
        if abs(dx)<0.004 and abs(dy)<0.004: break
        step(act(dx=dx,dy=dy)); n+=1
    for _ in range(300):
        dy=np.clip(f[1]-R('y'),-0.05,0.05)
        if abs(dy)<0.004: break
        step(act(dy=dy)); n+=1
    # push right until hook x >= 2.35
    for _ in range(400):
        if H('x')>=2.4: break
        f=fe()
        dy=np.clip(f[1]-R('y'),-0.02,0.02)
        step(act(dx=0.05,dy=dy)); n+=1
        if R('x')>3.2: break
    print("  after push hook",round(H('x'),3),round(H('y'),3),round(H('theta'),3),"robot",round(R('x'),3),round(R('y'),3),"steps",n,"vel",round(H('vx'),4),round(H('omega'),4))
    obs2,n2,ok=grasp_seq(env,obs)
    print("  grasp",ok,n2)
env.close()
