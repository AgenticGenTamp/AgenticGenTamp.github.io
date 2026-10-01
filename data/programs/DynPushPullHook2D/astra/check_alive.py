from env_client import make_env
import time
print('connect',flush=True)
e=make_env(); print('reset',flush=True)
s,i=e.reset(seed=101); print('ok',i,flush=True);e.close()
