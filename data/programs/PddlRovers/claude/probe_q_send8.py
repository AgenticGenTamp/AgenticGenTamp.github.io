from probe_vis_lib import *
import numpy as np, sys
seed=int(sys.argv[1]); yr=float(sys.argv[2])
env,obs,info=new_env(seed)
L=layout(obs); Lp=np.array([-1.9,-2.0])
cands=[(n,np.array([L[n]['x'],L[n]['y']])) for n in L if n.startswith('objective') and L[n]['x']>0.3]
on,O=cands[0]
obs,ok=nav(env,obs,O[0],max(-2.2,O[1]-1.5),i=0)
obs,_,_,_,_=st(env,op='calibrate',i=0); obs,_,_,_,_=st(env,op='image',i=0)
h=feats(obs,on)['have_image_rover0']
obs,ok=nav(env,obs,0.8,yr,i=0); p=pose(obs,0)[:2]
obs,_,_,_,_=st(env,op='send',i=0)
yc=p[1]+(p[0]/(p[0]+1.9))*(-2.0-p[1])
print("seed=%d have=%.0f pos=(%.3f,%.3f) d=%.3f cross_y=%.3f recv=%.0f"%(seed,h,p[0],p[1],np.linalg.norm(p-Lp),yc,feats(obs,on)['received_image']))
env.close()
