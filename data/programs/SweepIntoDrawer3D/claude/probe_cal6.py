import numpy as np
from env_client import make_env
from ctrl import move
from ik import ik
from ctrl import RDOWN
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env(); obs,_=env.reset(seed=0)
c0=obs[:80].reshape(5,16)[:,:3].copy()
obs,e,u=move(env,obs,[0.500,0.074,0.30],steps=250)
qd,_=ik(np.array([0.5,0.074,0.30]),RDOWN,obs[128:135])
print("q now",obs[128:135],"qd",qd,"jointerr",qd-obs[128:135],"used",u)
for z in np.arange(0.20,-0.05,-0.01):
    obs,e,u=move(env,obs,[0.500,0.074,z],steps=60)
    c=obs[:80].reshape(5,16)[:,:3]
    d=np.linalg.norm(c-c0,axis=1)
    qd,_=ik(np.array([0.5,0.074,z]),RDOWN,obs[128:135])
    print(f"z={z:.2f} jerr={np.round(qd-obs[128:135],3)} used={u} cubes={np.round(d,3)}")
    if d.max()>0.005: break
env.close()
