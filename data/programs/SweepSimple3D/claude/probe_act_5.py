import time
t=time.time()
from env_client import make_env
env=make_env()
print("made in %.1fs"%(time.time()-t),flush=True)
obs,_=env.reset(seed=0)
print("reset ok %.1fs"%(time.time()-t),flush=True)
env.close()
