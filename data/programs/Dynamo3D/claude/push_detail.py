import numpy as np, sys
from env_client import make_env
seed=int(sys.argv[1]); ang=float(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
names=sorted([n for n in obs.get_object_names() if n!='robot'])
cn=names[0]
def cf(o,f):
    return float(o.get(o.get_object_from_name(cn),f))
def cp(o): return np.array([cf(o,'x'),cf(o,'y')])
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'pos_base_x')),float(o.get(r,'pos_base_y')),float(o.get(r,'pos_base_rot'))])
c0=cp(obs); th=np.radians(ang); dirv=np.array([np.cos(th),np.sin(th)])
stage=c0-dirv*1.3
perp=np.array([-dirv[1],dirv[0]])
hist=[]
def rec(term=False):
    b=base(obs); c=cp(obs)
    hist.append((np.round(b,3).tolist(), np.round(c,3).tolist(), round(float(np.linalg.norm(b[:2]-c)),3), round(cf(obs,'z'),3), round(cf(obs,'qw'),3),round(cf(obs,'qz'),3)))
def goto(t):
    global obs
    for _ in range(300):
        b=base(obs)[:2]; d=t-b
        if np.linalg.norm(d)<0.06: return True
        step=np.clip(d,-0.1,0.1); nb=b+step*0.87
        if np.linalg.norm(nb-cp(obs))<0.95:
            v=nb-cp(obs); v/=np.linalg.norm(v); step=np.clip(cp(obs)+v*1.0+perp*0.3-b,-0.1,0.1)
        a=np.zeros(11,dtype=np.float32); a[:2]=step
        obs,r,term,trunc,info=env.step(a)
        if term: return False
    return True
if goto(stage):
    for i in range(300):
        a=np.zeros(11,dtype=np.float32); a[:2]=dirv*0.03
        obs,r,term,trunc,info=env.step(a)
        rec()
        if term:
            print("ang",ang,"seed",seed,"steps_push",i)
            for h in hist[-6:]: print("   ",h)
            break
    else: print("ang",ang,"no term")
else: print("ang",ang,"term staging")
