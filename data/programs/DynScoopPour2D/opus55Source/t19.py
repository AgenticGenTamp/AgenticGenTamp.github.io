import sys, numpy as np
from common import make_env
from approach import GeneratedApproach
sd=int(sys.argv[1]); c=int(sys.argv[2]); a0=int(sys.argv[3]); a1=int(sys.argv[4]); ev=int(sys.argv[5])
env=make_env(); s,info=env.reset(seed=sd,options={'object_count':c})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
for t in range(a1):
    a=ap.get_action(s); s,_,term,_,_=env.step(a)
    if t>=a0 and t%ev==0:
        ap._parse(s); r=ap.robot; h=ap.hook
        print(t,ap.phase,np.round(a,3),'R',round(r['x'],2),round(r['y'],2),round(r['theta'],2),'H',round(h['x'],2),round(h['y'],2),[(round(o[0],2),round(o[1],2)) for o in ap.objs][:3])
    if term: print('done',t); break
