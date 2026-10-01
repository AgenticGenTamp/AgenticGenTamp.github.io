from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(3)
L=layout(obs); Lp=np.array([-1.9,-2.0])
O=np.array([L['objective0']['x'],L['objective0']['y']])
obs,ok=nav(env,obs,O[0],O[1]-1.5,i=0)
obs,_,_,_,_=st(env,op='calibrate',i=0); obs,_,_,_,_=st(env,op='image',i=0)
print("have0",feats(obs,'objective0')['have_image_rover0'])
for tgt in [(1.6,-1.0),(1.0,-1.75),(0.5,-2.0),(0.24,-2.0),(0.24,-1.0)]:
    obs,ok=nav(env,obs,tgt[0],tgt[1],i=0); p=pose(obs,0)[:2]
    obs,_,_,_,_=st(env,op='send',i=0)
    print("pos=(%.2f,%.2f) d_lander=%.3f at_home=%.0f received=%.0f"%(p[0],p[1],np.linalg.norm(p-Lp),rf(obs,0)['at_home'],feats(obs,'objective0')['received_image']))
env.close()
