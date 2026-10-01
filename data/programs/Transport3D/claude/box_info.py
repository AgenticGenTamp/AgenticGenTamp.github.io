import numpy as np, kutil
from env_client import make_env
env=make_env()
for s in [0,1,2]:
    o,_=env.reset(seed=s)
    names=sorted(o.get_object_names())
    c=kutil.Ctl(env,o)
    out=[]
    for n in names:
        ob=o.get_object_from_name(n)
        try:
            p=[round(float(o.get(ob,f)),3) for f in ["pose_x","pose_y","pose_z"]]
            q=[round(float(o.get(ob,f)),3) for f in ["pose_qx","pose_qy","pose_qz","pose_qw"]]
        except Exception: p=q=None
        try: h=[round(float(o.get(ob,f)),3) for f in ["half_extent_x","half_extent_y","half_extent_z"]]
        except Exception: h=None
        out.append((n,p,q,h))
    print("seed",s)
    for r in out: print("  ",r)
env.close()
