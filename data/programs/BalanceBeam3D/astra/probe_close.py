from env_client import make_env
from approach import GeneratedApproach
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor

def run(mount):
    e=make_env()
    s,info=e.reset(seed=0)
    initial=s[:3].copy()
    p=GeneratedApproach(e.action_space,e.observation_space,{})
    p.mount=mount
    p.close=0.
    p.reset(s,info)
    hi=s[:3].copy()
    lo=s[:3].copy()
    for k in range(400):
        s,r,t,tr,info=e.step(p.get_action(s))
        hi=np.maximum(hi,s[:3]);lo=np.minimum(lo,s[:3])
        if k%100==99:
            print('CLOSE0',mount,k,'phase',p.phase,'large',np.round(s[:3],5),'gripper',s[26],flush=True)
        if t or tr:
            print('DONE',mount,k,r,t,tr,info,flush=True)
            break
    print('RESULT CLOSE0',mount,'initial',initial,'final',s[:3],'min',lo,'max',hi,'phase',p.phase,flush=True)
    e.close()

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run,map(float,sys.argv[1:] or ['.3','.4','.5'])))
