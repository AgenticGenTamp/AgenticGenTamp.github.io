from env_client import make_env
env = make_env()
for seed,oc in [(1,20),(3,4)]:
    try:
        obs, info = env.reset(seed=seed, options={'object_count':oc})
    except Exception as e:
        obs, info = env.reset(seed=seed); print('opt fail', e)
    names=sorted([n for n in obs.get_object_names() if n.startswith('cube')], key=lambda s:int(s[4:]))
    print(seed, info)
    for n in names:
        o=obs.get_object_from_name(n)
        print(' ',n, round(float(obs.get(o,'x')),3), round(float(obs.get(o,'y')),3), round(float(obs.get(o,'z')),3))
env.close()
