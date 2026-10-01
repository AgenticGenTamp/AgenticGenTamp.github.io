import numpy as np
from env_client import make_env
from lib_probe import *
env=make_env(); obs,info=env.reset(seed=0)
r=obs.get_object_from_name('robot')
bx,by=obs.get(r,'pos_base_x'),obs.get(r,'pos_base_y')
p0=obs.get_object_from_name('part0')
px,py=obs.get(p0,'pose_x'),obs.get(p0,'pose_y')
def probe(x,y,label):
    global obs
    obs,b=goto_xyz(env,obs,(x,y,0.40),base=(bx,by))
    z=0.40
    while z>0.05:
        z-=0.005
        obs,blocked=goto_xyz(env,obs,(x,y,z),base=(bx,by))
        if blocked:
            print(label,"blocked target z",round(z,3),"fk", round(float(fk(getq(obs))[2,3]),4)); return
    print(label,"reached 0.05")
probe(px, py-0.20, "empty table")   # away from part along y
probe(0.30, 0.0, "over rack center")
probe(px,py,"over part")
env.close()
