from env_client import make_env
env = make_env()
for seed in range(3,12):
    obs, info = env.reset(seed=seed, options={"object_count":20})
    print(seed, info)
    if info['object_count']==20:
        cs=[(n, round(obs.get(obs.get_object_from_name(n),'x'),3), round(obs.get(obs.get_object_from_name(n),'y'),3), round(obs.get(obs.get_object_from_name(n),'z'),3)) for n in sorted(obs.get_object_names(), key=lambda s:(len(s),s)) if n.startswith('cube')]
        print(cs); break
env.close()
