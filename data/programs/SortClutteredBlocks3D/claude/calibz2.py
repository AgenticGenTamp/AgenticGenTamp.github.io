import numpy as np, sys
from env_client import make_env
from ctrl import robot_state, action, cubes
from fk import ik, fk
RD = np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
TH = np.pi
R2 = np.array([[np.cos(TH),-np.sin(TH)],[np.sin(TH),np.cos(TH)]])
env=make_env(); obs,info=env.reset(seed=0)
b,q,g=robot_state(obs)
def goto(qt, bt, n):
    global b,q,g,obs
    for t in range(n):
        a=action(b,q,bt,qt,0.0)
        obs,r,te,tr,i=env.step(a); b,q,g=robot_state(obs)
R_ARM=0.45
def base_for(xy):
    return np.array([xy[0]-R2[0,0]*R_ARM, xy[1]-R2[1,0]*R_ARM, TH])
bt = base_for(np.array([0.3,0.3]))
qt = ik(np.array([R_ARM,0.0,-0.10]), RD, q)
goto(qt, bt, 140)
print('base', np.round(b,3), 'fk', np.round(fk(q)[:3,3],4))
prev=None
for h in np.arange(-0.12,-0.75,-0.02):
    qt = ik(np.array([R_ARM,0.0,h]), RD, q)
    goto(qt, bt, 20)
    p = fk(q)[:3,3]
    print(f"h={h:+.3f} z_reached={p[2]:+.4f} r={p[0]:+.3f} stall={h-p[2]:+.3f}")
env.close()
