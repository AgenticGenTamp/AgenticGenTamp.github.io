import math
import numpy as np
from env_client import make_env

def trial(fwd, lat=0.0):
    e=make_env(); s,_=e.reset(seed=0); typ=e.observation_space.get_type
    r=s.get_objects(typ('kin_robot'))[0]; b=s.get_objects(typ('target_block'))[0]
    th=float(s.get(r,'theta')); bx=float(s.get(b,'x')); by=float(s.get(b,'y'))
    tx=bx-fwd*math.cos(th)+lat*math.sin(th)
    ty=by-fwd*math.sin(th)-lat*math.cos(th)
    for i in range(25):
        dx=tx-float(s.get(r,'x')); dy=ty-float(s.get(r,'y'))
        a=np.array([np.clip(dx,-.049,.049),np.clip(dy,-.049,.049),0,0,0],float)
        s,*_=e.step(a)
        if abs(dx)<.003 and abs(dy)<.003: break
    pos=(float(s.get(b,'x')),float(s.get(b,'y')))
    hit=0
    for j in range(12):
        s,*_=e.step(np.array([0,0,0,0,-.019],float))
        if float(s.get(b,'held'))>.5: hit=j+1; break
    print('fwd',fwd,'move',round(pos[0]-bx,3),round(pos[1]-by,3),'held',hit,
          'gap',round(float(s.get(r,'finger_gap')),3),'final',round(float(s.get(b,'x')),3),round(float(s.get(b,'y')),3),flush=True)
    e.close()

for f in (.72,.68,.64,.60,.56,.52): trial(f)
