from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name('robot')
def r(obs): 
    s={f: round(obs.get(R,f),3) for f in ['x','y','theta','arm_joint','finger_gap']}
    for o in sorted(obs.data,key=lambda o:o.name):
        if o.name!='robot': s[o.name]=(round(obs.get(o,'x'),3),round(obs.get(o,'y'),3),round(obs.get(o,'theta'),2))
    return s
for a,n in [([0,0,0,0.099,0],3),([0,0,0,0.099,0],10),([0,0,0,-0.099,0],20),([0,0.049,0,0,0],40),([-0.049,0,0,0,0],30),([0,-0.049,0,0,0],40),([0.049,0,0,0,0],20),([0.049,0,0,0,0],20),([0.049,0,0,0,0],40)]:
    for i in range(n):
        try:
            obs,rew,term,trunc,info = env.step(np.array(a,dtype=float))
        except Exception as e:
            print("ERR", a, e); env.reset(seed=0); break
    print(a, r(obs), term)
env.close()
