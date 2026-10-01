import numpy as np
from env_client import make_env
from lib_probe import *
env=make_env(); obs,info=env.reset(seed=0)
r=obs.get_object_from_name('robot'); p0=obs.get_object_from_name('part0')
bx,by=obs.get(r,'pos_base_x'),obs.get(r,'pos_base_y')
px,py,pz=[obs.get(p0,f) for f in ['pose_x','pose_y','pose_z']]
print("descend over part (x,y)=",px,py)
z=0.35
obs,b=goto_xyz(env,obs,(px,py,z),base=(bx,by))
while z>0.0:
    z-=0.005
    obs,blocked=goto_xyz(env,obs,(px,py,z),base=(bx,by))
    if blocked:
        print("blocked at target z=",round(z,3),"actual fk z", round(fk(getq(obs))[2,3],4)); break
# now over empty table area
ex,ey = px, py+0.18
z=0.35
obs,b=goto_xyz(env,obs,(ex,ey,z),base=(bx,by))
while z>0.0:
    z-=0.005
    obs,blocked=goto_xyz(env,obs,(ex,ey,z),base=(bx,by))
    if blocked:
        print("empty area blocked at target z=",round(z,3),"actual", round(fk(getq(obs))[2,3],4)); break
env.close()
