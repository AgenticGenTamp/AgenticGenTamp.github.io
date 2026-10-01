import numpy as np
from env_client import make_env

def run(seed, policy, n=80):
    e=make_env(); s,_=e.reset(seed=seed)
    print('init',np.round(s,3))
    for t in range(n):
        a=np.asarray(policy(t,s),dtype=np.float32)
        ns,r,term,trunc,info=e.step(a)
        if t<5 or t%10==9 or np.linalg.norm(ns[20:22]-s[20:22])>.001 or np.linalg.norm(ns[9:11]-s[9:11])>.001:
            print(t+1,'a',np.round(a,2),'rob',np.round(ns[[0,1,2,4,6]],3),'hook',np.round(ns[[9,10,11]],3),'but',np.round(ns[20:22],3),'term',term)
        s=ns
        if term or trunc: break
    e.close()

if __name__=='__main__':
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else 'arm'
    seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
    if mode=='arm': run(seed,lambda t,s:[0,0,0,.1,0],10)
    elif mode=='rotate': run(seed,lambda t,s:[0,0,.196,0,0],20)
    elif mode=='direct':
        # point toward button, then approach/extend with vacuum
        phase=[0]
        def pol(t,s):
            th=np.arctan2(s[21]-s[1],s[20]-s[0]); err=(th-s[2]+np.pi)%(2*np.pi)-np.pi
            dist=np.linalg.norm(s[20:22]-s[:2])
            if abs(err)>.05: return [0,0,np.clip(err,-.196,.196),0,0]
            if dist>0.5: return [np.clip(s[20]-s[0],-.05,.05),np.clip(s[21]-s[1],-.05,.05),0,0,0]
            return [0,0,0,.1,1]
        run(seed,pol,80)
