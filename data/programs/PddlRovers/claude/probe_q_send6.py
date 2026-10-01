from probe_vis_lib import *
import numpy as np, sys
mode=sys.argv[1]
env,obs,info=new_env(3)
L=layout(obs); Lp=np.array([-1.9,-2.0])
O=np.array([L['objective0']['x'],L['objective0']['y']])
obs,ok=nav(env,obs,O[0],O[1]-1.5,i=0)
obs,_,_,_,_=st(env,op='calibrate',i=0); obs,_,_,_,_=st(env,op='image',i=0)
tgts={'home':[(1.0,-1.75),(1.0,-1.92),(1.25,-1.75),(1.0,-1.55)],
      'd329':[(0.24,0.5)],
      'd364':[(0.30,0.9),(0.24,0.5),(0.24,-0.6)]}[mode]
for tgt in tgts:
    obs,ok=nav(env,obs,tgt[0],tgt[1],i=0); p=pose(obs,0)[:2]
    obs,_,_,_,_=st(env,op='send',i=0)
    print("pos=(%.2f,%.2f) d=%.3f at_home=%.0f recv=%.0f"%(p[0],p[1],np.linalg.norm(p-Lp),rf(obs,0)['at_home'],feats(obs,'objective0')['received_image']))
    if feats(obs,'objective0')['received_image']>0.5: break
env.close()
