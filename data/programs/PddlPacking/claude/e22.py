from env_client import make_env
env=make_env()
for n in [1,4,5,6,8]:
    try:
        obs,info=env.reset(seed=1, options={"object_count":n})
        print(n, info, len([x for x in obs.get_object_names() if x.startswith("block")]))
    except Exception as e:
        print(n,"ERR",repr(e)[:200])
env.close()
