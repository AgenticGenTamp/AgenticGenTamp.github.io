import sys, time, pickle, numpy as np
from test_approach import make
import approach
from approach import GeneratedApproach
seed=int(sys.argv[1]); count=int(sys.argv[2])
env=make(); s,info=env.reset(seed=seed, options={'object_count':count})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s,info)
orig=ap._plan_to_any_inner
Q=[]
def w(chk,start,goals,tb,*a,**k):
    t=time.time(); res=orig(chk,start,goals,tb,*a,**k); dt=time.time()-t
    if dt>0.3: Q.append((chk,np.array(start),[np.array(g) for g in goals],tb,res is not None,dt))
    return res
ap._plan_to_any_inner=w
t0=time.time()
for t in range(300+150*count):
    a=ap.get_action(s); s,r,term,trunc,_=env.step(a)
    if term or time.time()-t0>60: break
pickle.dump(Q,open(f'logs/q_{seed}_{count}.pkl','wb'))
print(seed,count,len(Q),sum(q[4] for q in Q))
