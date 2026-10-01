import numpy as np, sys
from env_client import make_env

FEATS = ["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"]

def rollout(action, steps=50, seed=0, record_every=10):
    env = make_env()
    obs, info = env.reset(seed=seed)
    tf = {k.name:v for k,v in env.observation_space.type_features.items()}
    rob = [obs.get_object_from_name(n) for n in obs.get_object_names()
           if obs.get_object_from_name(n).type.name.startswith("mujoco_tidybot")][0]
    def vec(o):
        return np.array([float(o.get(rob,f)) for f in FEATS])
    hist=[vec(obs)]
    rews=[]; term=trunc=False; infos=[]
    for t in range(steps):
        obs,r,term,trunc,info = env.step(np.array(action,dtype=np.float32))
        hist.append(vec(obs)); rews.append(r); infos.append(info)
        if term or trunc: break
    env.close()
    return np.array(hist), np.array(rews), term, trunc, infos[-1] if infos else None

if __name__=="__main__":
    dim=int(sys.argv[1]); val=float(sys.argv[2]); grip=float(sys.argv[3]) if len(sys.argv)>3 else 0.0
    steps=int(sys.argv[4]) if len(sys.argv)>4 else 50
    a=[0.0]*10+[grip]
    if dim>=0: a[dim]=val
    h,r,term,trunc,info = rollout(a,steps=steps)
    np.set_printoptions(precision=4,suppress=True)
    print("DIM",dim,"val",val,"grip",grip,"steps",len(h)-1)
    print("feat        start      s1        s5        s10       s25       final     totaldelta")
    idx=[0,1,5,10,25,len(h)-1]
    for i,f in enumerate(FEATS):
        row=[h[min(j,len(h)-1),i] for j in idx]
        print(f"{f:14s}"+" ".join(f"{v:9.4f}" for v in row)+f" {h[-1,i]-h[0,i]:9.4f}")
    print("rew first5",np.round(r[:5],4),"sum",round(float(r.sum()),4))
    print("term",term,"trunc",trunc,"info",info)
