import numpy as np
from env_client import make_env
from approach import GeneratedApproach, world_fk, ik_solutions, grasp_R, dq_wrap
def run_to_stuck(env,seed,nsteps=20):
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    for t in range(nsteps):
        a=ap.get_action(obs); obs,_,_,_,_=env.step(a)
    return ap,obs
def servo(env,ap,obs,qt,n=8):
    for k in range(n):
        r=ap._robot(obs); dq=dq_wrap(qt-r["q"])
        if np.max(np.abs(dq))<1e-7: return obs,False
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(dq,-0.2,0.2)
        prev=np.concatenate([r["base"],r["q"]])
        obs,_,_,_,_=env.step(a)
        r2=ap._robot(obs)
        if np.max(np.abs(np.concatenate([r2["base"],r2["q"]])-prev))<1e-9: return obs,True
    return obs,False
env=make_env()
for (db,dz,grip) in [(0.0,0.03,0),(0.0,0.03,-1),(0.0,0.0,0),(0.0,0.07,0),(0.02,0.03,0),(-0.02,0.03,0)]:
    ap,obs=run_to_stuck(env,1,19)
    r=ap._robot(obs); base=r["base"]; q=r["q"]
    g0=ap._blocks(obs)["green0"]; d=ap.dir; R=grasp_R(np.arctan2(d[1],d[0]))
    perp=np.array([-d[1],d[0],0.0])
    if grip:
        a=np.zeros(11,dtype=np.float32); a[10]=grip; obs,_,_,_,_=env.step(a)
    pg=np.array([g0[0]-d[0]*0.02,g0[1]-d[1]*0.02,g0[2]+dz])+perp*db
    tool=world_fk(q,base)[0]
    t0=np.dot(pg-tool,d)
    stop=None
    for t in np.arange(t0,-0.001,-0.01):
        p=pg-d*max(t,0.0)
        sols=ik_solutions(p,R,base,q,seeds=3,iters=100)
        if not sols: stop=("ikfail",t); break
        obs,rej=servo(env,ap,obs,sols[0],8)
        if rej: stop=("rej",t); break
        q=ap._robot(obs)["q"]
    if stop is None:
        a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,_,_,_,_=env.step(a)
        print(f"db={db} dz={dz} grip={grip}: GRASP={ap._robot(obs)['holding']}")
    else:
        print(f"db={db} dz={dz} grip={grip}: stopped {stop[0]} at dist={stop[1]:.3f} tool={np.round(world_fk(ap._robot(obs)['q'],base)[0],3)}")
env.close()
