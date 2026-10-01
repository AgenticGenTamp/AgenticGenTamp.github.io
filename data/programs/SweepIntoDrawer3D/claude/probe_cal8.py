import numpy as np
from env_client import make_env
from ctrl import move
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env(); obs,_=env.reset(seed=0)
c0=obs[:80].reshape(5,16)[:,:3].copy()
obs,e,u=move(env,obs,[0.500,-0.12,0.30],steps=300,grip=0.0)
obs,e,u=move(env,obs,[0.500,-0.12,0.205],steps=80,grip=0.0)
print("start err",e)
prev=c0
for y in np.arange(-0.10,0.30,0.02):
    obs,e,u=move(env,obs,[0.50,y,0.205],steps=40,grip=0.0)
    c=obs[:80].reshape(5,16)[:,:3]
    d=np.linalg.norm(c-prev,axis=1)
    if d.max()>0.003:
        print(f"y={y:.2f} err={e:.3f} moved={np.round(d,3)} newpos={np.round(c[d.argmax()],3)}")
    else:
        print(f"y={y:.2f} err={e:.3f}")
    prev=c.copy()
env.close()
