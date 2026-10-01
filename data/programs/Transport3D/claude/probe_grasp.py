import numpy as np
from probe_util import *
from probe_collide import drive, uparm

env=make_env(); obs,_=env.reset(seed=3)
O=objs(obs); c=O["cube0"]; print("cube0",tuple(round(x,3) for x in c))
obs=uparm(env,obs)
# put the arm column (base + 0.12*forward, rot=0) at various offsets behind the cube
res=[]
for off in [0.5,0.6,0.7]:
    for _ in range(1):
        pass
obs=drive(env,obs,c[0]-0.6,c[1],order="yx")
print("base",fmt(rs(obs)[:3]))
# bend j2 forward until blocked; record
cur=rs(obs)
for i in range(40):
    o2,_,_,_,_=env.step(act(j2=0.1))
    if np.allclose(rs(o2),cur): break
    obs=o2;cur=rs(o2)
print("j2 blocked at",round(cur[4],3),"after",i,"steps; pose",fmt(cur[3:10],2))
# open then close gripper here
for g in [1.0,-1.0]:
    obs,_,_,_,_=env.step(act(g=g))
    print(" g",g,"finger",round(rs(obs)[10],4),"grasp",round(rs(obs)[11],4))
# scan base x closer/farther and try closing each time
for _ in range(8):
    o2,_,_,_,_=env.step(act(bx=0.05))
    if np.allclose(rs(o2),rs(obs)): print("  base blocked"); break
    obs=o2
    obs,_,_,_,_=env.step(act(g=-1.0))
    s=rs(obs); cu=objs(obs)["cube0"]
    print("  basex",round(s[0],3),"finger",round(s[10],3),"grasp",round(s[11],3),"cube_z",round(cu[2],3))
env.close()
