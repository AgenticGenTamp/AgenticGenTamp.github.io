from env_client import make_env
env = make_env()
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("seed", seed, info, [(n, round(obs.get(obs.get_object_from_name(n),'x'),3), round(obs.get(obs.get_object_from_name(n),'y'),3), round(obs.get(obs.get_object_from_name(n),'z'),3)) for n in sorted(obs.get_object_names()) if n.startswith('cube')])
env.close()
