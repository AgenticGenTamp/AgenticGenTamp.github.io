from env_client import make_env
env=make_env()
for seed in [25,31,53,50,13]:
    obs,info=env.reset(seed=seed)
    for n in sorted(obs.get_object_names()):
        if n.startswith("obstacle"):
            o=obs.get_object_from_name(n)
            print(seed,n,"x",round(float(obs.get(o,"x")),3),"y",round(float(obs.get(o,"y")),4),"h",round(float(obs.get(o,"height")),4))
env.close()
