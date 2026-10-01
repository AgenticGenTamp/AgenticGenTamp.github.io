import numpy as np
from probe_util import *
env=make_env()
obs,_=env.reset(seed=0)
print("objs",{k:tuple(round(x,3) for x in v) for k,v in objs(obs).items()})
# 1. rotate base then command +x
for _ in range(5):
    obs,_,_,_,_=env.step(act(br=0.2))
print("after 5 br=0.2:",fmt(rs(obs)[:3]))
b=rs(obs)[:3]
obs,_,_,_,_=env.step(act(bx=0.2))
print("after bx=0.2 delta:",fmt(rs(obs)[:3]-b))
b=rs(obs)[:3]
obs,_,_,_,_=env.step(act(by=0.2))
print("after by=0.2 delta:",fmt(rs(obs)[:3]-b))
# rotation range
cur=rs(obs)
for i in range(200):
    o2,_,_,_,_=env.step(act(br=0.2))
    if np.allclose(rs(o2),cur): break
    obs=o2;cur=rs(o2)
print("max rot after",i,"steps:",round(cur[2],4))
for i in range(400):
    o2,_,_,_,_=env.step(act(br=-0.2))
    if np.allclose(rs(o2),cur): break
    obs=o2;cur=rs(o2)
print("min rot after",i,"steps:",round(cur[2],4))
env.close()

# world bounds on x,y (drive far in each direction from a fresh env)
for (dx,dy,name) in [(0.2,0,"+x"),(-0.2,0,"-x"),(0,0.2,"+y"),(0,-0.2,"-y")]:
    env=make_env(); obs,_=env.reset(seed=1)
    # move off the x axis first for +x to avoid table? no: keep as is
    cur=rs(obs)
    for i in range(150):
        o2,_,_,_,_=env.step(act(bx=dx,by=dy))
        if np.allclose(rs(o2),cur): break
        obs=o2;cur=rs(o2)
    # refine
    for step in [0.05,0.01,0.002]:
        for _ in range(6):
            o2,_,_,_,_=env.step(act(bx=np.sign(dx)*step,by=np.sign(dy)*step))
            if np.allclose(rs(o2),cur): break
            obs=o2;cur=rs(o2)
    print(name,"stopped at",fmt(cur[:2]),"steps",i,"objs",{k:tuple(round(x,2) for x in v) for k,v in objs(obs).items()},flush=True)
    env.close()
