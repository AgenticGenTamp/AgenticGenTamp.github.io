from probe_vis_lib import *
import numpy as np, sys
yr=float(sys.argv[1])
env,obs,info=new_env(3)
L=layout(obs); Lp=np.array([-1.9,-2.0])
O=np.array([L['objective0']['x'],L['objective0']['y']])
obs,ok=nav(env,obs,O[0],O[1]-1.5,i=0)
obs,_,_,_,_=st(env,op='calibrate',i=0); obs,_,_,_,_=st(env,op='image',i=0)
obs,ok=nav(env,obs,0.8,yr,i=0); p=pose(obs,0)[:2]
obs,_,_,_,_=st(env,op='send',i=0)
yc=p[1]+(p[0]/(p[0]+1.9))*(-2.0-p[1])
print("pos=(%.3f,%.3f) d=%.3f cross_y=%.3f recv=%.0f"%(p[0],p[1],np.linalg.norm(p-Lp),yc,feats(obs,'objective0')['received_image']))
env.close()
