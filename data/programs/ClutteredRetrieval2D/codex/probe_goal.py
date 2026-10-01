import math
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

env=make_env(); s,info=env.reset(seed=1,options={'object_count':0})
p=GeneratedApproach(env.action_space,env.observation_space,{}) ; p.reset(s,info)
t={q.name:q for q in env.observation_space.types}
def one(n): return list(s.get_objects(t[n]))[0]
def v(o,f): return float(s.get(o,f))
for i in range(300):
    s,r,d,tr,inf=env.step(p.get_action(s))
    b,g=one('target_block'),one('target_region')
    if p.phase=='transport' and math.hypot(v(b,'x')-v(g,'x'),v(b,'y')-v(g,'y'))<.006: break
print('staged',i,v(b,'x')-v(g,'x'),v(b,'y')-v(g,'y'),v(b,'theta')-v(g,'theta'))
steps=0
for iy in range(21):
    yy=-.20+iy*.02
    xs=range(21) if iy%2==0 else range(20,-1,-1)
    for ix in xs:
        xx=-.20+ix*.02
        for q in range(10):
            b,g=one('target_block'),one('target_region')
            dx=v(g,'x')+xx-v(b,'x');dy=v(g,'y')+yy-v(b,'y')
            if max(abs(dx),abs(dy))<.003:break
            a=np.array([np.clip(dx,-.02,.02),np.clip(dy,-.02,.02),0,0,1],np.float32)
            s,r,d,tr,inf=env.step(a);steps+=1
            if d: print('HIT',xx,yy,'actual',v(one('target_block'),'x')-v(one('target_region'),'x'),v(one('target_block'),'y')-v(one('target_region'),'y'),'steps',steps);env.close();raise SystemExit
print('no hit',steps);env.close()
