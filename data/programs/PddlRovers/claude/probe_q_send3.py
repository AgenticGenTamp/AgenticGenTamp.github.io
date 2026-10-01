from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(3)
L=layout(obs); Lp=np.array([-1.9,-2.0])
O=np.array([L['objective2']['x'],L['objective2']['y']])
obs,ok=nav(env,obs,-1.7,0.5,i=1); print("nav",ok,pose(obs,1))
obs,_,_,_,_=st(env,op='calibrate',i=1); print("calib",rf(obs,1)['calibrated'])
obs,_,_,_,_=st(env,op='image',i=1); print("have1",feats(obs,'objective2')['have_image_rover1'])
u=np.array([-1.2,2.0])-Lp; u/=np.linalg.norm(u)
start=Lp+4.30*u
obs,ok=nav(env,obs,start[0],start[1],i=1); print("nav2",ok,pose(obs,1),np.linalg.norm(pose(obs,1)[:2]-Lp))
for k in range(60):
    p=pose(obs,1)[:2]; d=np.linalg.norm(p-Lp)
    obs,_,_,_,_=st(env,op='send',i=1)
    r=feats(obs,'objective2')['received_image']
    print("d=%.4f pos=(%.3f,%.3f) recv=%.0f"%(d,p[0],p[1],r))
    if r>0.5: break
    tgt=p-0.02*u
    obs,_=goto(env,obs,tgt[0],tgt[1],i=1,tol=0.003,maxit=6)
env.close()
