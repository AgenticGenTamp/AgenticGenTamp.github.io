from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(3)
L=layout(obs)
O=np.array([L['objective0']['x'],L['objective0']['y']]); Z=L['objective0']['z']
print("obj0",O,Z)
# approach from due south
obs,ok=nav(env,obs,O[0],O[1]-2.25,i=0)
print("nav",ok,pose(obs,0))
prev=None
for k in range(40):
    p=pose(obs,0)[:2]
    d=np.linalg.norm(p-O)
    obs,v=vis_test(env,obs,0)
    print("d2=%.4f d3=%.4f vis=%d theta=%.2f"%(d,np.hypot(d,Z),v,pose(obs,0)[2]))
    if v: break
    obs,_,_,_,_=st(env,dy=0.01,i=0)
    if np.linalg.norm(pose(obs,0)[:2]-p)<1e-9:
        print("stuck"); break
# theta dependence: rotate to face away/toward, test calibrate
for th in [0.0,1.57,-1.57,3.14]:
    p=pose(obs,0)
    dth=th-p[2]
    while abs(dth)>0.01:
        s=float(np.clip(dth,-0.4,0.4)); obs,_,_,_,_=st(env,dth=s,i=0); dth=th-pose(obs,0)[2]
    obs,v=vis_test(env,obs,0)
    print("theta=%.2f vis=%d"%(pose(obs,0)[2],v))
# persistence while driving
obs,_,_,_,_=st(env,op='calibrate',i=0)
print("calib",rf(obs,0)['calibrated'])
for k in range(3):
    obs,_,_,_,_=st(env,dy=-0.2,i=0)
    print("  drove, calibrated=",rf(obs,0)['calibrated'])
obs,_,_,_,_=st(env,dth=0.4,i=0)
print("  rotated, calibrated=",rf(obs,0)['calibrated'])
d=np.linalg.norm(pose(obs,0)[:2]-O)
obs,_,_,_,_=st(env,op='image',i=0)
print("image at d=%.3f -> have0=%.0f calib=%.0f"%(d,feats(obs,'objective0')['have_image_rover0'],rf(obs,0)['calibrated']))
env.close()
