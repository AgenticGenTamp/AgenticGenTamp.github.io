import numpy as np
from env_client import make_env
from lib_probe import *
env=make_env(); obs,info=env.reset(seed=0)
r=obs.get_object_from_name('robot'); bx,by=obs.get(r,'pos_base_x'),obs.get(r,'pos_base_y')
p0=obs.get_object_from_name('part0'); px,py=obs.get(p0,'pose_x'),obs.get(p0,'pose_y')
print("part start",px,py)
# approach beside part then push toward -y
obs,b=goto_xyz(env,obs,(px,py+0.20,0.30),base=(bx,by)); print("above side",b)
obs,b=goto_xyz(env,obs,(px,py+0.20,0.245),base=(bx,by)); print("low",b, float(fk(getq(obs))[2,3]))
for i in range(12):
    yy = py+0.20-0.02*(i+1)
    obs,b=goto_xyz(env,obs,(px,yy,0.245),base=(bx,by))
    print(round(yy,3),"blocked",b,"part y",round(float(obs.get(p0,'pose_y')),4), "fkz", round(float(fk(getq(obs))[2,3]),3))
    if b: break
env.close()
