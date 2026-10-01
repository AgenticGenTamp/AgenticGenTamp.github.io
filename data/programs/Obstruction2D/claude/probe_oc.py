from env_client import make_env
env=make_env()
for oc in [3,5]:
    try:
        obs,info=env.reset(seed=1, options={'object_count':oc})
        print(oc, info, len(obs.get_object_names()))
    except Exception as e:
        print(oc,"ERR",repr(e)[:200])
env.close()
