import numpy as np, time
from kinder.envs.kinematic2d.clutteredretrieval2d import ObjectCentricClutteredRetrieval2DEnv as E
from approach import *
e=E(num_obstructions=20); s,_=e.reset(seed=2)
ap=GeneratedApproach(None,None,{}); r,rects=ap._parse(s); m=ap._robot_model(r)
chk=Checker(m,ap._obstacles(rects,set()),[])
q=ap._q(r)
for B in [1,8,64]:
    Q=np.repeat(q[None],B,0)
    t=time.time()
    for _ in range(2000): batch_free(chk,Q)
    print(B,(time.time()-t)/2000*1e6,"us")
