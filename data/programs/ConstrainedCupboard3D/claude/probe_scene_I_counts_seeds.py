import numpy as np, math
from env_client import make_env
env=make_env()
for n in [4,5,6]:
    ysets=set(); xs=[];ys=[];zs=set()
    for s in range(6):
        obs,info=env.reset(seed=s, options={'object_count':n})
        names=obs.get_object_names()
        cup=tuple(sorted(round(float(obs.data[obs.get_object_from_name(c)][1]),3) for c in names if c.startswith('cupboard')))
        cx=set(round(float(obs.data[obs.get_object_from_name(c)][0]),3) for c in names if c.startswith('cupboard'))
        cz=set(round(float(obs.data[obs.get_object_from_name(c)][2]),3) for c in names if c.startswith('cupboard'))
        ysets.add((cup,tuple(sorted(cx)),tuple(sorted(cz))))
        for c in names:
            if c.startswith('cuboid'):
                d=obs.data[obs.get_object_from_name(c)]
                xs.append(float(d[0])); ys.append(float(d[1])); zs.add(round(float(d[2]),4))
        _,r,_,_,_=env.step(np.zeros(11,dtype=np.float32))
    print(f"n={n} r0={r} distinct cupboard layouts across 6 seeds: {len(ysets)}")
    for u in ysets: print("   ys",u[0],"x",u[1],"z",u[2])
    print(f"   cuboid x[{min(xs):.3f},{max(xs):.3f}] y[{min(ys):.3f},{max(ys):.3f}] z {sorted(zs)}")
env.close()
