from env_client import make_env
for seed in range(10):
 e=make_env();s,_=e.reset(seed=seed,options={'object_count':1});p=s.get_object_from_name('part0')
 fs=s.type_features[p.type]
 print(seed,p.type.name,{f:s.get(p,f) for f in fs if f in ('triangle_type','pose_x','pose_y')});e.close()
