from env_client import make_env
import numpy as np
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
for g in [1.0,0.0]:
    env=make_env(); obs,_=env.reset(seed=0)
    if g==0.0:
        a=np.zeros(11,np.float32); a[10]=1.0
        for _ in range(60): obs,*_=env.step(a)
        print("pre-open pos_gripper=%.4f"%rob(obs)['pos_gripper'])
    a=np.zeros(11,np.float32); a[10]=g
    tr=[]
    for i in range(80):
        obs,*_=env.step(a); tr.append(rob(obs)['pos_gripper'])
    print("gripper cmd",g,[round(v,4) for v in tr[:20]])
    print("   ...",[round(v,4) for v in tr[20:80:5]])
    fin=tr[-1]
    k=next((i for i,v in enumerate(tr) if abs(v-fin)<0.01),None)
    print("   final=%.4f  steps to within 0.01 of final: %s"%(fin,k))
    env.close()
# intermediate values
env=make_env()
for g in [0.25,0.5,0.75]:
    obs,_=env.reset(seed=0); a=np.zeros(11,np.float32); a[10]=g
    for _ in range(60): obs,*_=env.step(a)
    print("cmd %.2f -> pos_gripper %.4f"%(g,rob(obs)['pos_gripper']))
env.close()
