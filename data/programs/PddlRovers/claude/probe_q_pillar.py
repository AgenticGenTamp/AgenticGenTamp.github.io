from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(3)
L=layout(obs)
O=np.array([L['objective0']['x'],L['objective0']['y']])
P=np.array([L['obstacle10']['x'],L['obstacle10']['y']])
u=(P-O)/np.linalg.norm(P-O)
perp=np.array([-u[1],u[0]])
def segdist(a,b,c):
    v=b-a; t=np.clip(np.dot(c-a,v)/np.dot(v,v),0,1); return np.linalg.norm(a+t*v-c)
print("obj",O,"pillar",P,"d_obj_pil",np.linalg.norm(P-O))
for off in np.arange(0.0,0.62,0.04):
    tgt=O+1.90*u+off*perp
    obs,ok=nav(env,obs,tgt[0],tgt[1],i=0)
    p=pose(obs,0)[:2]
    if not ok:
        print("offset %.2f nav fail at"%off,p); continue
    d=np.linalg.norm(p-O); sd=segdist(p,O,P)
    obs,v=vis_test(env,obs,0)
    print("off=%.2f pos=(%.3f,%.3f) d_obj=%.3f segdist_pillar=%.3f vis=%d"%(off,p[0],p[1],d,sd,v))
    if v: break
env.close()
