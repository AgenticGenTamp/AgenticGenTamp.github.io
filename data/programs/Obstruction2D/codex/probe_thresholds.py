import numpy as np
from env_client import make_env

def one(offset, step=.01, vac=1.0, seed=0):
    e=make_env(); s,_=e.reset(seed=seed); T=e.observation_space
    r=s.get_objects(T.get_type("crv_robot"))[0]; b=s.get_objects(T.get_type("target_block"))[0]
    bx=float(s.get(b,"x")); target=bx+offset
    # vacuum on while horizontally aligning
    for _ in range(30):
        dx=np.clip(target-float(s.get(r,"x")),-.05,.05)
        s,*_=e.step(np.array([dx,0,0,0,vac],np.float32))
        if abs(dx)<1e-6: break
    last=float(s.get(r,"y")); stop=None
    for i in range(100):
        by0=float(s.get(b,"y")); y0=float(s.get(r,"y"))
        s,*_=e.step(np.array([0,-step,0,0,vac],np.float32))
        y1=float(s.get(r,"y"))
        if abs(y1-y0)<1e-7: stop=(i,y1,by0); break
    b0=(float(s.get(b,"x")),float(s.get(b,"y")))
    s,*_=e.step(np.array([0,.05,0,0,vac],np.float32))
    b1=(float(s.get(b,"x")),float(s.get(b,"y")))
    print("off",offset,"step",step,"vac",vac,"stop",stop,"b",tuple(round(x,6) for x in b0),"=>",tuple(round(x,6) for x in b1),"held",abs(b1[1]-b0[1])>.02)
    e.close()

for off in [0,.03,.045,.05,.052,.055,.06,.08,-.045,-.052]:
    one(off)
for st in [.05,.02,.01,.005,.001]: one(0,st)
