from env_client import make_env
import numpy as np
np.set_printoptions(precision=4, suppress=True)
exec(open('probe_geom_1.py').read().split('def probe')[0])
obs,_=env.reset(seed=12)
w,lh,lv=obs[12],obs[13],obs[14]
ths=[obs[2]]; trace=[]
for it in range(60):
    pose=obs[0:3].copy(); Rm=R(pose[2]); cen=pose[:2]+Rm@np.array([0,-lv/2])
    start=pose[:2]+Rm@np.array([lh/2-0.08,-w-0.13])
    route(start,cen,1.05)
    d=Rm@np.array([0,1.0])
    for k in range(8):
        step(d*0.04); ths.append(obs[2]); trace.append((obs[2],r if False else 0))
    # back off
    for k in range(3): step(-d*0.04)
    if it%5==0: print(it, obs[0:3])
ths=np.array(ths)
print('theta min/max',ths.min(),ths.max())
jumps=np.where(np.abs(np.diff(ths))>1)[0]
for j in jumps[:6]: print('jump',ths[j],'->',ths[j+1])
print('unwrapped total rotation', np.unwrap(ths)[-1]-ths[0])
