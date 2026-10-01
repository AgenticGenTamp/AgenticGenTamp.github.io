from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(20)
L=layout(obs)
O=np.array([L['objective0']['x'],L['objective0']['y']]); P=np.array([L['obstacle7']['x'],L['obstacle7']['y']])
dop=np.linalg.norm(P-O); u=(P-O)/dop
print("obj",np.round(O,3),"pillar",np.round(P,3),"d",round(dop,3))
def segd(a,b,c):
    v=b-a; t=np.clip(np.dot(c-a,v)/np.dot(v,v),0,1); return np.linalg.norm(a+t*v-c)
for D in (1.90,1.60,1.30,1.05):
    tg=O+D*u
    obs,ok=nav(env,obs,tg[0],tg[1],i=0); p=pose(obs,0)[:2]
    d=np.linalg.norm(p-O); sd=segd(p,O,P); t=(d-dop)/d
    others={n:round(segd(p,O,np.array([L[n]['x'],L[n]['y']])),2) for n in L if n.startswith('obstacle') and n!='obstacle7'}
    obs,v=vis_test(env,obs,0)
    mn=min(others.values())
    print("D=%.2f pos=(%.3f,%.3f) d=%.3f segdist_p7=%.3f t_pillar=%.3f vis=%d min_other_segdist=%.2f"%(D,p[0],p[1],d,sd,t,v,mn))
env.close()
