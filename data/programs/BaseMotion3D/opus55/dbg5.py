import numpy as np
from env_client import make_env
env=make_env()
def go(o,goal,steps=200,minsc=0.005):
    sc=1.0
    for _ in range(steps):
        d=goal-o[:2]
        if np.linalg.norm(d)<1e-4: break
        a=np.zeros(11); a[:2]=np.clip(d,-0.4*sc,0.4*sc)
        b=o[:3].copy(); o,r,te,*_=env.step(a)
        if te: return o,True
        if np.linalg.norm(o[:3]-b)<1e-7:
            sc*=0.5
            if sc<minsc: break
    return o,False
for x in []:
    o,_=env.reset(seed=0)
    o,_=go(o,np.array([x,-1.0])); o,_=go(o,np.array([x,-3.0]))
    print(round(x,2),np.round(o[:2],3))
print('rot sweep')
def rot_to(o,r):
    for _ in range(20):
        dr=(r-o[2]+np.pi)%(2*np.pi)-np.pi
        if abs(dr)<1e-3: break
        a=np.zeros(11); a[2]=np.clip(dr,-0.4,0.4); o,*_=env.step(a)
    return o
for r in np.arange(-3.0,3.2,0.2):
    o,_=env.reset(seed=0)
    o,_=go(o,np.array([0.4,-1.0])); o=rot_to(o,r); o,_=go(o,np.array([0.4,-3.0]))
    print(round(r,2),np.round(o[:3],3))
