import numpy as np, sys
from env_client import make_env
seed=1; T=np.array([0.959,-0.009])
env=make_env(); obs,info=env.reset(seed=seed)
cn=sorted([n for n in obs.get_object_names() if n!='robot'])[0]
def cp(o): 
    c=o.get_object_from_name(cn); return np.array([float(o.get(c,'x')),float(o.get(c,'y'))])
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'pos_base_x')),float(o.get(r,'pos_base_y'))])
c0=cp(obs); dirv=(T-c0)/np.linalg.norm(T-c0)
stage=c0-dirv*1.3
perp=np.array([-dirv[1],dirv[0]])
def goto(t):
    global obs
    for _ in range(400):
        b=base(obs); d=t-b
        if np.linalg.norm(d)<0.05: return True
        step=np.clip(d,-0.1,0.1); nb=b+step*0.87
        if np.linalg.norm(nb-cp(obs))<0.95:
            v=nb-cp(obs); v/=np.linalg.norm(v); step=np.clip(cp(obs)+v*1.0+perp*0.35-b,-0.1,0.1)
        a=np.zeros(11,dtype=np.float32); a[:2]=step
        obs,r,term,trunc,info=env.step(a)
        if abs(r+1)>1e-6: print("REW",round(r,4),np.round(cp(obs),3),flush=True)
        if term: print("TERM staging"); return False
    return True
print("chair",np.round(c0,3),"dist to T",round(float(np.linalg.norm(T-c0)),3))
if goto(stage):
    for i in range(300):
        a=np.zeros(11,dtype=np.float32); a[:2]=dirv*0.03
        obs,r,term,trunc,info=env.step(a)
        d=np.linalg.norm(cp(obs)-T)
        if abs(r+1)>1e-6 or i%5==0: print(i,round(r,4),term,"chair",np.round(cp(obs),3),"dT=%.3f"%d,flush=True)
        if term: print("TERM chair",np.round(cp(obs),3),"dT=%.3f"%d); break
