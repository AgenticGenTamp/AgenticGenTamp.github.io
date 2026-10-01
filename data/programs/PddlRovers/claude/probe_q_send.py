from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(3)
L=layout(obs); Lp=np.array([-1.9,-2.0])
O=np.array([L['objective0']['x'],L['objective0']['y']])
obs,ok=nav(env,obs,O[0],O[1]-1.5,i=0); print("nav obj",ok,pose(obs,0))
obs,_,_,_,_=st(env,op='calibrate',i=0); print("calib",rf(obs,0)['calibrated'])
obs,_,_,_,_=st(env,op='image',i=0)
print("have0",feats(obs,'objective0')['have_image_rover0'],"recv",feats(obs,'objective0')['received_image'])
u=np.array([1.2,-1.0])-Lp; u/=np.linalg.norm(u)
start=Lp+4.30*u
obs,ok=nav(env,obs,start[0],start[1],i=0); print("nav start",ok,pose(obs,0),np.linalg.norm(pose(obs,0)[:2]-Lp))
for k in range(30):
    p=pose(obs,0)[:2]; d=np.linalg.norm(p-Lp)
    obs,_,_,_,_=st(env,op='send',i=0)
    r=feats(obs,'objective0')['received_image']
    print("dist_lander=%.4f pos=(%.3f,%.3f) received=%.0f"%(d,p[0],p[1],r))
    if r>0.5: break
    tgt=p-0.02*u
    obs,_=goto(env,obs,tgt[0],tgt[1],i=0,tol=0.003,maxit=6)
env.close()
