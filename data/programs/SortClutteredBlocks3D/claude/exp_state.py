from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
try:
    s=env.get_state(); print('get_state ok', type(s))
    try: print('len',len(s))
    except Exception as e: print(e)
except Exception as e: print('get_state FAIL', repr(e)[:200])
try:
    env.set_state(s); print('set_state ok')
except Exception as e: print('set_state FAIL', repr(e)[:200])
env.close()
