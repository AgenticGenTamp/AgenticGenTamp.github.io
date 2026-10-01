from env_client import make_env
E=make_env()
for seed in range(8):
 s,i=E.reset(seed=seed)
 print('seed',seed,i)
 for t in ['shelf','target_block']:
  for o in s.get_objects(E.observation_space.get_type(t)):
   print(o.name, [round(s.get(o,f),3) for f in (['x1','y1','width1','height1'] if t=='shelf' else ['x','y','theta'])])
E.close()
