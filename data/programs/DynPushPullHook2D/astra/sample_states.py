from env_client import make_env
import json
E=make_env(); results=[]
for n in [1,2,3,4,5]:
 for seed in range(3):
  s,i=E.reset(seed=seed,options={'object_count':n})
  t=s.get_objects(E.observation_space.get_type('target_block'))[0]
  h=s.get_objects(E.observation_space.get_type('hook'))[0]
  print(seed,n,flush=True); results.append({'seed':seed,'count':n,'info':i,'target':{f:s.get(t,f) for f in ['x','y','width','height','theta']},'hook':{f:s.get(h,f) for f in ['x','y','theta','length_side1','length_side2']}})
print(json.dumps(results,indent=1))
E.close()
