from env_client import make_env
E=make_env()
mins={};maxs={}
for seed in range(100):
 s,_=E.reset(seed=seed)
 for name in s.get_object_names():
  o=s.get_object_from_name(name)
  for f in ['x','y']:
   key=('button' if name.startswith('button') else name)+f
   v=s.get(o,f);mins[key]=min(v,mins.get(key,v));maxs[key]=max(v,maxs.get(key,v))
print(mins);print(maxs);E.close()
