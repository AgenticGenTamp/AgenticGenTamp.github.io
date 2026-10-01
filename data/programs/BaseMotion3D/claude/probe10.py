import numpy as np
from env_client import make_env
env=make_env()
def goto(o, gx, gy, step=0.1):
    for k in range(200):
        a=np.zeros(11)
        dx=np.clip(gx-o[0],-step,step); dy=np.clip(gy-o[1],-step,step)
        if abs(dx)<1e-7 and abs(dy)<1e-7: break
        a[0],a[1]=dx,dy
        o2,r,t,tr,_=env.step(a)
        if np.linalg.norm(o2[:2]-o[:2])<1e-9: 
            o=o2; break
        o=o2
    return o
o,_=env.reset(seed=0)
o=goto(o,0.0,-1.5)
print("at",o[:2])
for d in [0.1,0.2,0.3,0.35,0.39,0.4]:
    e=make_env(); oo,_=e.reset(seed=0); oo=goto(oo,0.0,-1.5)
    a=np.zeros(11); a[1]=-d
    oo2,r,t,tr,_=e.step(a)
    print("step",d,"->",oo2[1], "moved", abs(oo2[1]-oo[1])>1e-7)
    e.close()
# test swept: go far around obstacle? try jump over thin wall: from (0,-1.5) step (0.4,-0.4)... 
