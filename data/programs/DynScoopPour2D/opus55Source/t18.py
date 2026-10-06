import sys
from common import make_env
from approach import GeneratedApproach
sd=int(sys.argv[1]); c=int(sys.argv[2])
env=make_env(); s,info=env.reset(seed=sd,options={'object_count':c})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
ph=None
for t in range(300+150*c):
    s,_,term,_,_=env.step(ap.get_action(s))
    if ap.phase!=ph:
        ph=ap.phase; ap._parse(s); print(t,ph,{k:round(v,2) for k,v in ap.robot.items() if k in 'xy'},[(round(o[0],2),round(o[1],2)) for o in ap.objs])
    if term: print('done',t); break
