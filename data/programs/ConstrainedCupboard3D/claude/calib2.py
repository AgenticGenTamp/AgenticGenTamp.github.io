import sys, numpy as np, kin
from env_client import make_env

MX, MY, MZ, TOOL = (float(x) for x in sys.argv[1:5])
def rq(o):
    r=o.get_object_from_name('robot'); f=o.type_features[r.type]; d=o.data[r]
    return np.array([d[f.index('pos_arm_joint%d'%(i+1))] for i in range(7)]), np.array([d[0],d[1],d[2]])
env=make_env(); o,info=env.reset(seed=0)
q,b=rq(o)
rods=[n for n in o.get_object_names() if n.startswith('cuboid')]
tgt=min(rods,key=lambda n:np.hypot(*(o.data[o.get_object_from_name(n)][:2]-b[:2])))
rd=o.data[o.get_object_from_name(tgt)]
print("rod",tgt,np.round(rd[:3],3),"base",np.round(b,3))
# drive base to rod_x-0.45, rod_y, rot 0  (world-frame assumption)
goal=np.array([rd[0]-0.45, rd[1], 0.0])
for i in range(120):
    q,b=rq(o)
    a=np.zeros(11,dtype=np.float32)
    a[:3]=np.clip((goal-b)*1.0,-0.1,0.1)
    a[10]=1.0
    o,r,t,tr,inf=env.step(a)
    if np.linalg.norm(goal-b)<0.005: break
q,b=rq(o)
print("after drive base",np.round(b,4),"steps",i)
rd=o.data[o.get_object_from_name(tgt)]
print("rod now",np.round(rd[:3],3))
