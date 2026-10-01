import numpy as np, sys
from env_client import make_env
env=make_env(); o,info=env.reset(seed=0)   # 3 cuboids, dividers at +-0.05,0.15,0.25
# candidate targets: y slot centers 0, +-0.1, +-0.2 ; place 3 rods at x=2.0 along x-axis, different z
qw,qz=np.cos(np.pi/4),np.sin(np.pi/4)  # yaw 90deg -> body y along world x
cands=[(2.0,0.0,0.015),(2.0,-0.1,0.30),(2.0,0.1,0.60)]
for i,(x,y,z) in enumerate(cands):
    ob=o.get_object_from_name('cuboid_%d'%i)
    d=o.data[ob]; d[0],d[1],d[2]=x,y,z; d[3],d[4],d[5],d[6]=qw,0,0,qz
p=env.render_state(state=o,label='tgt'); print(p)
env.close()
