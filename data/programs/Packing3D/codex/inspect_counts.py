from env_client import make_env
for seed in range(10):
 e=make_env();s,_=e.reset(seed=seed,options={'object_count':2});out=[]
 for n in sorted(x for x in s.get_object_names() if x.startswith('part')):
  o=s.get_object_from_name(n);out.append((n,o.type.name,s.get(o,'triangle_type') if o.type.name.endswith('Triangle') else None,round(s.get(o,'pose_y'),3)))
 print(seed,out);e.close()
