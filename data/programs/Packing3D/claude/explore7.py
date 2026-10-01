import numpy as np
from env_client import make_env
from lib_probe import *
env=make_env(); obs,info=env.reset(seed=0)
r=obs.get_object_from_name('robot')
bx,by=obs.get(r,'pos_base_x'),obs.get(r,'pos_base_y')
p0=obs.get_object_from_name('part0'); px,py=obs.get(p0,'pose_x'),obs.get(p0,'pose_y')
HOME=np.array([0.0,-0.35,-3.1416,-2.5,0.0,-0.87,1.5708])
def minz(x,y):
    global obs
    obs,_=move_to(env,obs,HOME)
    obs,b0=goto_xyz(env,obs,(x,y,0.45),base=(bx,by))
    lo,hi=0.05,0.45
    # binary search on reachable z (assume monotone)
    for _ in range(8):
        mid=(lo+hi)/2
        obs,blocked=goto_xyz(env,obs,(x,y,mid),base=(bx,by))
        zz=float(fk(getq(obs))[2,3])
        if blocked or abs(zz-mid)>0.01: lo=mid
        else: hi=mid
    return hi, b0
for (x,y,lbl) in [(px,py,'over part0'),(px,py-0.2,'empty'),(0.3,0.0,'rack center'),(0.3,0.2,'empty2'),(0.15,0.0,'near rack front'),(0.45,0.0,'far')]:
    z,b0=minz(x,y)
    print(lbl,(round(x,3),round(y,3)),"min fk z ~",round(z,3),"approach blocked",b0)
env.close()
