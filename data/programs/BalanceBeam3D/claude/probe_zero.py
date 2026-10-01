import numpy as np
from env_client import make_env
env=make_env()
for seed in [0,1,2]:
    obs,info=env.reset(seed=seed); o0=np.asarray(obs,float)
    rs=[]
    a=np.zeros(11,dtype=np.float32)
    last=None
    for t in range(200):
        obs,r,term,trunc,info=env.step(a)
        rs.append(r)
        if term or trunc:
            print("ended",t,term,trunc); break
        last=np.asarray(obs,float)
    rs=np.array(rs)
    print(f"seed{seed} r[0:5]={np.round(rs[:5],5)} unique={np.unique(np.round(rs,6))[:6]} sum={rs.sum():.4f}")
    d=np.abs(last-o0)
    big=np.where(d>1e-3)[0]
    print("  moved idx>1e-3:",big.tolist()[:30])
    print("  info@step:",info)
    print("  lb",np.round(last[0:3],4),"ss",np.round(last[38:41],4),np.round(last[41:45],4),"sb1",np.round(last[54:57],4),"sb2",np.round(last[70:73],4))
    print("  gripper/arm now",np.round(last[19:27],3))
