import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=200)
env = make_env()
obs, info = env.reset(seed=0)
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def st(a):
    global obs
    obs, r, te, tr, _ = env.step(np.array(a, dtype=np.float32))
    return r, te
def goto(x,y,th,arm=None,vac=0,n=200):
    for i in range(n):
        dx=np.clip(x-obs[0],-.05,.05); dy=np.clip(y-obs[1],-.05,.05); dt=np.clip(wrap(th-obs[2]),-.196,.196)
        da=0 if arm is None else np.clip(arm-obs[4],-.1,.1)
        if max(abs(dx),abs(dy),abs(dt),abs(da))<1e-4: return i
        prev=obs.copy(); st([dx,dy,dt,da,vac])
        if np.allclose(prev[:5],obs[:5]): print('stuck',obs[:5]); return -1
    return n
hx,hy,ht=obs[9:12]
u=-np.array([np.cos(ht),np.sin(ht)]); n=np.array([u[1],-u[0]])
if n[0]<0: n=-n
P=np.array([hx,hy])+u*1.0
print('P',P,'n',n)
R=P+n*0.35
th=np.arctan2(-n[1],-n[0])
goto(R[0],R[1],th,arm=0.1)
print('at',obs[:9])
# approach slowly
for i in range(40):
    prev=obs.copy()
    st([-n[0]*0.01,-n[1]*0.01,0,0,0])
    if np.allclose(prev[:2],obs[:2]): print('contact base at',i); break
print(obs[:9], 'dist to line', np.dot(obs[:2]-P,n))
for i in range(3):
    st([0,0,0,0.1,0]); print('arm',obs[4])
st([0,0,0,0,1]); print('vac', obs[:12])
for i in range(5):
    st([n[0]*0.05,n[1]*0.05,0,0,1]); print('move', obs[:2], obs[9:12])
for i in range(3):
    st([0,0,0.1,0,1]); print('rot', obs[:3], obs[9:12])
for i in range(3):
    st([0,0,0,0.05,1]); print('arm', obs[:5], obs[9:12])
st([0,0,0,0,0]); print('vac off', obs[:7], obs[9:12])
st([0.05,0,0,0,0]); print('move', obs[:7], obs[9:12])
