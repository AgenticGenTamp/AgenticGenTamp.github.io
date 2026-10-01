import numpy as np, sys
from env_client import make_env
cnt=3
env=make_env(); o,info=env.reset(seed=0,options={'object_count':cnt})
# upright rods (long axis = world z): rotate body y to world z -> rotation about x by 90deg
qw,qx=np.cos(np.pi/4),np.sin(np.pi/4)
specs=[(1.60,-0.2,0.15),(1.60,0.0,0.15),(1.60,0.2,0.15)]
for i,(x,y,z) in enumerate(specs):
    d=o.data[o.get_object_from_name('cuboid_%d'%i)]
    d[0],d[1],d[2]=x,y,z; d[3],d[4],d[5],d[6]=qw,qx,0,0
print(env.render_state(state=o,label='ruler'))
env.close()
