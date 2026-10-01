import numpy as np, threading
from env_client import make_env
from open_drawer import attempt, RROLL
out={}; lock=threading.Lock()
def job(wy,wz):
    env=make_env(); obs,_=env.reset(seed=0)
    try:
        obs,rep=attempt(env,obs,wy,wz)
    except Exception as e:
        rep={'err':str(e)[:80]}
    env.close()
    with lock: out[(wy,wz)]=rep
jobs=[(wy,wz) for wz in [0.24,0.32,0.40] for wy in [-0.25,0.00]]
ths=[threading.Thread(target=job,args=j) for j in jobs]
[t.start() for t in ths]; [t.join() for t in ths]
for k in sorted(out): print(k,out[k])
