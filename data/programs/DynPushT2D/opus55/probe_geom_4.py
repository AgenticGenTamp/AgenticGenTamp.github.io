from env_client import make_env
import numpy as np
np.set_printoptions(precision=4, suppress=True)
exec(open('probe_geom_3.py').read().split('obs,_=env.reset')[0])
obs,_=env.reset(seed=12)
w,lh,lv=obs[12],obs[13],obs[14]
for it in range(2):
    pose=obs[0:3].copy(); Rm=R(pose[2]); cen=pose[:2]+Rm@np.array([0,-lv/2])
    V=verts(obs); start=np.array([V[:,0].max()+0.2, cen[1]])
    route(start,cen,1.05)
    for k in range(60):
        p=obs[16:18].copy(); b=obs[0:3].copy()
        step(np.array([-0.04,0.0]))
        if it==1 and k>=36: print(k,'robot dx %.4f'%(obs[16]-p[0]),'block d',obs[0:3]-b,'minx %.4f'%verts(obs)[:,0].min())
