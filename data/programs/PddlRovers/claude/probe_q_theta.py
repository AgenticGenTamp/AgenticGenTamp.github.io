from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(3)
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
for dist in (1.95,1.60,1.0):
    obs,ok=nav(env,obs,O[0],O[1]-dist,i=0)
    p=pose(obs,0)[:2]; d=np.linalg.norm(p-O)
    res=[]
    for th in np.arange(-3.0,3.2,0.5):
        pp=pose(obs,0); dth=th-pp[2]
        while abs(dth)>0.02:
            obs,_,_,_,_=st(env,dth=float(np.clip(dth,-0.4,0.4)),i=0); dth=th-pose(obs,0)[2]
        obs,v=vis_test(env,obs,0)
        res.append((round(pose(obs,0)[2],2),int(v)))
    print("d=%.3f pos=%.3f,%.3f bearing_to_obj=%.2f"%(d,p[0],p[1],np.arctan2(O[1]-p[1],O[0]-p[0])), res)
# determinism repeat at one theta
for r in range(4):
    obs,v=vis_test(env,obs,0); print("repeat",int(v))
env.close()
