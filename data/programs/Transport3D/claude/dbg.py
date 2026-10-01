import sys, numpy as np
from env_client import make_env
import approach
from approach import GeneratedApproach
seed=int(sys.argv[1]) if len(sys.argv)>1 else 1
oc=int(sys.argv[2]) if len(sys.argv)>2 else None
env=make_env()
kw={'options':{'object_count':oc}} if oc else {}
o,info=env.reset(seed=seed,**kw)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(o,info)
prev=None
for i in range(1000):
    a=ap.get_action(o)
    key=(ap.cur.name if ap.cur else None, ap.phase)
    if key!=prev:
        b,q,g=ap._rob(o)
        print(i,key,"base",np.round(b,2),"ee",np.round(ap._ee(b,q),3),"g",g)
        prev=key
    o2,r,t,tr,info=env.step(a)
    same = all(np.allclose(o.get(o.get_object_from_name(n), f), o2.get(o2.get_object_from_name(n), f)) for n in o.get_object_names() for f in [])
    o=o2
    if t: print("TERMINATED at",i); break
print("final:")
for n in sorted(o.get_object_names()):
    ob=o.get_object_from_name(n)
    try: print(" ",n,np.round([float(o.get(ob,f)) for f in ["pose_x","pose_y","pose_z"]],3))
    except: pass
env.close()
