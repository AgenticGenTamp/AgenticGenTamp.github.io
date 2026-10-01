from env_client import make_env
env = make_env()
for seed in [2,3,4]:
    obs, info = env.reset(seed=seed)
    for i in range(1,5):
        o=obs.get_object_from_name(f'cube{i}')
        print(seed, i, round(float(obs.get(o,'x')),3), round(float(obs.get(o,'y')),3))
env.close()
