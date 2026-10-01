import sys
from env_client import make_env
from approach import GeneratedApproach

seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
limit=int(sys.argv[2]) if len(sys.argv)>2 else 1000
e=make_env(); s,i=e.reset(seed=seed); p=GeneratedApproach(e.action_space,e.observation_space,e.make_primitives());p.reset(s,i)
for k in range(limit):
    a=p.get_action(s); s,r,t,tr,i=e.step(a)
    if k%50==49:
        rs=sorted(s.get_objects(e.observation_space.get_type('rover')),key=lambda x:x.name)
        goals=[]
        for j in range(2):
            goals.append(p._vantage(s,j,s.get_object_from_name(p.assignments[j][0])) if p.assignments[j] else None)
        print(k+1,[(round(s.get(x,'x'),2),round(s.get(x,'y'),2),s.get(x,'store_full'),s.get(x,'calibrated'),s.get(x,'at_home')) for x in rs],p.assignments,p.sample_targets,p.mode,'a',a.tolist(),'g',goals)
    if t: print('DONE',k+1);break
for typ in ('objective','sample'):
 for o in s.get_objects(e.observation_space.get_type(typ)):
  print(o.name,[s.get(o,f) for f in e.observation_space.type_features[e.observation_space.get_type(typ)]])
e.close()
