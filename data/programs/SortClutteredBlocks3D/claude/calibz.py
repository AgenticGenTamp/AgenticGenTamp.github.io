import numpy as np, sys
from env_client import make_env
from ctrl import robot_state, action, cubes
from fk import ik, fk
RD = np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
env=make_env(); obs,info=env.reset(seed=0)
b,q,g=robot_state(obs); b0=b.copy()
def goto(qt, n):
    global b,q,g,obs
    for t in range(n):
        a=action(b,q,b0,qt,0.0)
        obs,r,te,tr,i=env.step(a); b,q,g=robot_state(obs)
qt = ik(np.array([0.45,0.0,-0.05]), RD, q)
goto(qt, 120)
print('reached', np.round(fk(q)[:3,3],4), 'err', np.round(((qt-q+np.pi)%(2*np.pi)-np.pi),3))
for h in np.arange(-0.06,-0.42,-0.02):
    qt = ik(np.array([0.45,0.0,h]), RD, q)
    goto(qt, 25)
    p = fk(q)[:3,3]
    print(f"h={h:+.3f} reached={np.round(p,4)} jerr={np.round(((qt-q+np.pi)%(2*np.pi)-np.pi),3)}")
print('cubes', {k:np.round(v,3) for k,v in cubes(obs).items()})
env.close()
