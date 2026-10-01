import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, world_fk, ik_solutions, grasp_R, dq_wrap
import approach as A
sd=1
env=make_env(); obs,info=env.reset(seed=sd)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for t in range(18):   # up to blocker dumped + start green pick
    a=ap.get_action(obs); obs,_,_,_,_=env.step(a)
r=ap._robot(obs); print("after dump, hold",r["holding"],"tool",np.round(world_fk(r["q"],r["base"])[0],3))
g0=ap._blocks(obs)["green0"]; d=ap.dir; R=grasp_R(np.arctan2(d[1],d[0]))
base=r["base"]
def servo(obs,qt,n=6):
    for k in range(n):
        r=ap._robot(obs); dq=dq_wrap(qt-r["q"])
        if np.max(np.abs(dq))<1e-6: return obs,False
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(dq,-0.2,0.2)
        prev=np.concatenate([r["base"],r["q"]])
        obs,_,_,_,_=env.step(a)
        r2=ap._robot(obs)
        if np.max(np.abs(np.concatenate([r2["base"],r2["q"]])-prev))<1e-8: return obs,True
    return obs,False
for (db,dz) in [(0.0,0.03),(0.0,0.0),(0.01,0.03),(-0.01,0.03),(0.0,0.06),(0.0,-0.01)]:
    obs,info=env.reset(seed=sd)
    ap2=GeneratedApproach(env.action_space,env.observation_space,{}); ap2.reset(obs,info)
    for t in range(16):
        a=ap2.get_action(obs); obs,_,_,_,_=env.step(a)
    r=ap2._robot(obs); base=r["base"]
    perp=np.array([-d[1],d[0],0.0])
    pg=np.array([g0[0]-d[0]*0.02,g0[1]-d[1]*0.02,g0[2]+dz])+perp*db
    q=r["q"]; stop=None
    for t in np.arange(0.20,-0.001,-0.01):
        p=pg-d*t
        sols=ik_solutions(p,R,base,q,seeds=4,iters=90)
        if not sols: stop=("ikfail",t); break
        qn=sols[0]
        obs,rej=servo(obs,qn,8)
        if rej: stop=("rej",t); break
        q=ap2._robot(obs)["q"]
    if stop is None:
        a=np.zeros(11,dtype=np.float32); a[10]=-1.0
        obs,_,_,_,_=env.step(a)
        print(f"db={db} dz={dz}: reached grasp, grasp_active={ap2._robot(obs)['holding']}")
    else:
        print(f"db={db} dz={dz}: stopped {stop[0]} at t={stop[1]:.2f}")
env.close()
