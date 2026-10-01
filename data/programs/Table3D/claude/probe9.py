import numpy as np
from env_client import make_env
J=["joint_%d"%i for i in range(1,8)]
def getq(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in J])
for j in range(7):
    for sgn in [1,-1]:
        env=make_env(); obs,_=env.reset(seed=0); q=getq(obs)
        last=q[j]
        for _ in range(60):
            a=np.zeros(11); a[3+j]=sgn*0.2
            o2,*_=env.step(a); qn=getq(o2)
            if abs(qn[j]-q[j])<1e-9:
                break
            q=qn
        print("joint",j+1,"dir",sgn,"reached %.3f"%q[j])
        env.close()
