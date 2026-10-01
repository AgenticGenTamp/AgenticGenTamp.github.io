import numpy as np
from probe_vis_lib import *
env,obs,info=new_env(1)
O=feats(obs,'objective0'); op=np.array([O['x'],O['y']])
start=pose(obs,0)[:2]
u=(start-op); u=u/np.linalg.norm(u)
tgt=op+u*2.08
obs,ok=goto(env,obs,tgt[0],tgt[1],i=0,tol=0.002)
print("start probe",pose(obs,0), "d=",np.linalg.norm(pose(obs,0)[:2]-op), ok)
d_prev=None
for k in range(120):
    p=pose(obs,0)[:2]; d=np.linalg.norm(p-op)
    obs,_,_,_,_=st(env,op='calibrate',i=0)
    if rf(obs,0)['calibrated']>0.5:
        print("THRESH: first success at 2d=%.5f (prev fail %.5f) 3d=%.5f"%(d,d_prev,np.sqrt(d*d+O['z']**2)))
        break
    d_prev=d
    obs,_,_,_,_=st(env,dx=u[0]*-0.001,dy=u[1]*-0.001,i=0)
else: print("no success, d=",np.linalg.norm(pose(obs,0)[:2]-op))
# heading: rotate a lot and confirm calibrate still ok after image clears
print("calibrated",rf(obs,0)['calibrated'])
obs,_,_,_,_=st(env,op='image',i=0)
print("after image: calibrated=%s have0_obj0=%s have0_obj1=%s"%(rf(obs,0)['calibrated'],feats(obs,'objective0')['have_image_rover0'],feats(obs,'objective1')['have_image_rover0']))
for th in [0,1,2]:
    for _ in range(4): obs,_,_,_,_=st(env,dth=0.4,i=0)
    obs,_,_,_,_=st(env,op='calibrate',i=0)
    print("theta=%.2f calibrated=%s"%(pose(obs,0)[2],rf(obs,0)['calibrated']))
    obs,_,_,_,_=st(env,op='image',i=0)
    print("   image-> calib=%s have0_obj0=%s have0_obj1=%s"%(rf(obs,0)['calibrated'],feats(obs,'objective0')['have_image_rover0'],feats(obs,'objective1')['have_image_rover0']))
env.close()
