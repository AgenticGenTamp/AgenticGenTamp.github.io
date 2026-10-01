import numpy as np, sys
from env_client import make_env
seed=int(sys.argv[1]); idx=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
names=sorted([n for n in obs.get_object_names() if n!='robot'])
cn=names[idx]
def cp(o,n):
    c=o.get_object_from_name(n); return np.array([float(o.get(c,'x')),float(o.get(c,'y'))])
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'pos_base_x')),float(o.get(r,'pos_base_y'))])
tgt=cp(obs,cn)
others=[n for n in names if n!=cn]
for i in range(400):
    b=base(obs); d=tgt-b
    step=np.clip(d,-0.1,0.1)
    nb=b+step*0.87
    # avoid other chairs
    for n in others:
        o=cp(obs,n)
        if np.linalg.norm(nb-o)<0.95:
            v=nb-o; v/=np.linalg.norm(v)
            perp=np.array([-v[1],v[0]])
            if np.dot(perp,d)<0: perp=-perp
            step=np.clip(o+v*1.0+perp*0.4-b,-0.1,0.1)
    a=np.zeros(11,dtype=np.float32); a[:2]=step
    obs,r,term,trunc,info=env.step(a)
    if abs(r+1)>1e-6: print("REW",round(r,4),flush=True)
    if term:
        print("seed %d target %s TERM at step %d robot %s chairs %s"%(seed,cn,i,np.round(base(obs),3),{n:np.round(cp(obs,n),2).tolist() for n in names}))
        break
else:
    print("seed %d target %s NO TERM robot %s"%(seed,cn,np.round(base(obs),3)))
