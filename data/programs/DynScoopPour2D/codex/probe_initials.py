from env_client import make_env
for seed in range(10):
 e=make_env(); s,i=e.reset(seed=seed); h=s.get_object_from_name('hook'); r=s.get_object_from_name('robot')
 print(seed,i, 'H',*[round(s.get(h,f),3) for f in ('x','y','theta')], 'R',*[round(s.get(r,f),3) for f in ('x','y','theta')]); e.close()
