from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(3)
L=layout(obs); Lp=np.array([-1.9,-2.0])
O=np.array([L['objective0']['x'],L['objective0']['y']])
obs,ok=nav(env,obs,O[0],O[1]-1.5,i=0)
obs,_,_,_,_=st(env,op='calibrate',i=0); obs,_,_,_,_=st(env,op='image',i=0)
for tgt in [(0.24,-0.5),(0.24,-1.4),(0.30,-2.20),(0.9,-2.2),(0.7,-2.2)]:
    obs,ok=nav(env,obs,tgt[0],tgt[1],i=0); p=pose(obs,0)[:2]
    obs,_,_,_,_=st(env,op='send',i=0)
    print("pos=(%.2f,%.2f) d=%.3f recv=%.0f"%(p[0],p[1],np.linalg.norm(p-Lp),feats(obs,'objective0')['received_image']))
env.close()
