from env_client import make_env
from kin import *
import sys
np.set_printoptions(precision=3,suppress=True)
J=[f'joint_{i}' for i in range(1,8)]
def getq(o):
    r=o.get_object_from_name('robot'); return np.array([o.get(r,n) for n in J])
def step_to(env,obs,qt,grip=0.0):
    for _ in range(50):
        q=getq(obs); d=qt-q
        if np.abs(d).max()<1e-6: return obs,True
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(d,-0.4,0.4); a[10]=grip
        obs2,*_=env.step(a)
        if np.abs(getq(obs2)-q).max()<1e-9: return obs2,False
        obs=obs2
    return obs,False
env=make_env()
for (x,y) in [(0.5,0.0),(0.648,-0.173),(0.431,0.235)]:
    obs,_=env.reset(seed=0)
    q=getq(obs); z=0.2
    while z>-0.4:
        qt,e=ik(q,np.array([x,y,z]),down_R(np.pi/2))
        obs,ok=step_to(env,obs,qt)
        if not ok:
            print('blocked at target z',round(z,3),'x,y',x,y,'q',getq(obs), 'fk', fk(getq(obs))[:3,3]); break
        q=getq(obs); z-=0.005
env.close()
