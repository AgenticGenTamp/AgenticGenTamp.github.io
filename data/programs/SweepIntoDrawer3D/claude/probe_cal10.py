import numpy as np
from env_client import make_env
from ctrl import move, RFWD
np.set_printoptions(precision=3,suppress=True,linewidth=200)
for zf in [0.10,0.06,0.02,-0.02]:
    env=make_env(); obs,_=env.reset(seed=0)
    c0=obs[:80].reshape(5,16)[:,:3].copy()
    obs,e,u=move(env,obs,[0.20,0.074,zf],R=RFWD,steps=300,grip=0.0)
    print(f"--- fk_z={zf} approach err={e:.3f} used={u} q={np.round(obs[128:135],2)}")
    prev=obs[:80].reshape(5,16)[:,:3].copy()
    for x in np.arange(0.22,0.56,0.02):
        obs,e,u=move(env,obs,[x,0.074,zf],R=RFWD,steps=40,grip=0.0)
        c=obs[:80].reshape(5,16)[:,:3]
        d=np.linalg.norm(c-prev,axis=1)
        if d.max()>0.004 or e>0.05:
            print(f"  x={x:.2f} err={e:.3f} moved={np.round(d,3)}")
        prev=c.copy()
    env.close()
