import numpy as np
from env_client import make_env

def trial(xoff, vac=1.0, extend=False, seed=0):
    e=make_env(); s,_=e.reset(seed=seed); T=e.observation_space
    r=s.get_objects(T.get_type("crv_robot"))[0]; b=s.get_objects(T.get_type("target_block"))[0]
    bx=float(s.get(b,"x")); tx=bx+xoff
    for _ in range(30):
        dx=float(np.clip(tx-float(s.get(r,"x")),-.05,.05))
        s,*_=e.step(np.array([dx,0,0,0,vac],np.float32))
        if abs(dx)<1e-6: break
    if extend: s,*_=e.step(np.array([0,0,0,.1,vac],np.float32))
    for i in range(100):
        y0=float(s.get(r,"y")); s,*_=e.step(np.array([0,-.01,0,0,vac],np.float32))
        if abs(float(s.get(r,"y"))-y0)<1e-7: break
    ry=float(s.get(r,"y")); rx=float(s.get(r,"x")); aj=float(s.get(r,"arm_joint")); b0=(float(s.get(b,"x")),float(s.get(b,"y")))
    s,*_=e.step(np.array([0,.05,0,0,vac],np.float32)); b1=(float(s.get(b,"x")),float(s.get(b,"y")))
    held=abs(b1[1]-b0[1])>.02
    # release while translating another .05 up
    s,*_=e.step(np.array([0,.05,0,0,0],np.float32)); b2=(float(s.get(b,"x")),float(s.get(b,"y")))
    print("off %.3f vac %.2f ext %s actualx %.6f stopY %.6f joint %.3f held %s after_on %s after_off %s"%(xoff,vac,extend,rx,ry,aj,held,tuple(round(x,6) for x in b1),tuple(round(x,6) for x in b2)))
    e.close()

for x in [-.04,-.036,-.035,-.034,-.02,0,.099,.1,.105,.13,.135,.136,.14]: trial(x)
for ext in [False,True]: trial(.05,1,ext)
for v in [0,.01,.1,.49,.5,.51,.9,1]: trial(.05,v,False)
