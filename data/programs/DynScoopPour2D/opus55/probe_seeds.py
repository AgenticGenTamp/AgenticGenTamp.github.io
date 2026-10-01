from env_client import make_env
env = make_env()
for seed in range(20):
    obs, info = env.reset(seed=seed)
    r = obs.get_object_from_name('robot'); h = obs.get_object_from_name('hook')
    sm = [n for n in obs.get_object_names() if n.startswith('small')]
    xs = [obs.get(obs.get_object_from_name(n),'x') for n in sm]
    print(seed, info['object_count'], len(sm), 'robot', [round(obs.get(r,f),2) for f in ('x','y','theta')], 'hook', [round(obs.get(h,f),2) for f in ('x','y','theta')], 'objx', round(min(xs),2), round(max(xs),2))
env.close()
