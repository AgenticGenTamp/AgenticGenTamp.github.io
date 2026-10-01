from env_client import make_env
import numpy as np
np.set_printoptions(precision=4, suppress=True)
exec(open('probe_geom_1.py').read().split('def probe')[0])
def verts(o):
    w,lh,lv=o[12],o[13],o[14]; Rm=R(o[2])
    loc=np.array([[-lh/2,0],[lh/2,0],[lh/2,-w],[-lh/2,-w],[-w/2,-w-lv],[w/2,-w-lv]])
    return o[0:2]+loc@Rm.T
obs,_=env.reset(seed=12)
w,lh,lv=obs[12],obs[13],obs[14]
for it in range(40):
    pose=obs[0:3].copy(); Rm=R(pose[2]); cen=pose[:2]+Rm@np.array([0,-lv/2])
    V=verts(obs); 
    # start at right of block at centroid height
    start=np.array([V[:,0].max()+0.2, cen[1]])
    route(start,cen,1.05)
    for k in range(40):
        step(np.array([-0.04,0.0]))
        if obs[16]<=0.1001: break
    V=verts(obs)
    print(it,'pose',obs[0:3],'min vert x %.4f'%V[:,0].min(),'robot',obs[16:18])
    if obs[16]<=0.2: break
