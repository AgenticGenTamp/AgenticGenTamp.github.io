import time,sys
from test_approach import make
from approach import GeneratedApproach
env=make(); s,info=env.reset(seed=int(sys.argv[1]),options={'object_count':int(sys.argv[2])})
ap=GeneratedApproach(env.action_space,None,{}); ap.reset(s,info)
ta=te=0
for t in range(5000):
    t0=time.time(); a=ap.get_action(s); t1=time.time(); s,r,term,trunc,_=env.step(a); t2=time.time()
    ta+=t1-t0; te+=t2-t1
    if term or trunc: break
print("steps",t+1,"term",term,"agent",round(ta,1),"env",round(te,1))
