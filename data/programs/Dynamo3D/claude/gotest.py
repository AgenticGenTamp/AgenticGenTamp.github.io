import numpy as np, sys
from env_client import make_env
seed=int(sys.argv[1]); G=np.array([float(sys.argv[2]),float(sys.argv[3])])
env=make_env(); obs,info=env.reset(seed=seed)
names=sorted([n for n in obs.get_object_names() if n!='robot'])
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'pos_base_x')),float(o.get(r,'pos_base_y'))])
def cp(o,n):
    c=o.get_object_from_name(n); return np.array([float(o.get(c,'x')),float(o.get(c,'y'))])
a0=np.zeros(11,dtype=np.float32)
for _ in range(40): obs,r,term,trunc,info=env.step(a0)
print("seed",seed,"chairs",{n:np.round(cp(obs,n),2).tolist() for n in names})
for i in range(300):
    b=base(obs); d=G-b
    if np.linalg.norm(d)<0.02:
        print("arrived at G, no term after", i,"steps; robot",np.round(b,3)); break
    a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d,-0.1,0.1)
    obs,r,term,trunc,info=env.step(a)
    if term:
        print("TERM seed",seed,"robot",np.round(base(obs),3),"dist to G",round(float(np.linalg.norm(base(obs)-G)),3)); break
else:
    print("loop end robot",np.round(base(obs),3))
# hover around
if not term:
    for k,off in enumerate([(0.15,0),(0,0.15),(-0.15,0),(0,-0.15),(0.3,0),(0,0.3),(-0.3,0),(0,-0.3)]):
        t=G+np.array(off)
        for i in range(60):
            b=base(obs); d=t-b
            if np.linalg.norm(d)<0.02: break
            a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d,-0.1,0.1)
            obs,r,term,trunc,info=env.step(a)
            if term: print("TERM at offset",off,np.round(base(obs),3)); sys.exit()
    print("no term anywhere near G")
