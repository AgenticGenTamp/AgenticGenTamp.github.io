from env_client import make_env
env = make_env()
for oc in [2,5,10,15]:
  try:
    obs, info = env.reset(seed=0, options={'object_count':oc})
    sh = obs.get_object_from_name('shelf')
    blocks=[n for n in obs.get_object_names() if n.startswith('block')]
    ins=[n for n in blocks if obs.get(obs.get_object_from_name(n),'y')>2.6]
    print(oc, info, len(blocks), len(ins), obs.get(sh,'width1'))
  except Exception as e: print(oc, 'err', str(e)[:200])
