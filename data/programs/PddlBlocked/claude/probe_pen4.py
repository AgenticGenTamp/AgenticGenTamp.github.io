import numpy as np
from env_client import make_env
from approach import GeneratedApproach, world_fk, ik_solutions, grasp_R, dq_wrap
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
for dz in [0.07,0.06,0.05,0.04]:
    obs,info=env.reset(seed=1)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    for t in range(19):
        a=ap.get_action(obs); obs,_,_,_,_=env.step(a)
    r=ap._robot(obs); base=r["base"]; q=r["q"]
    g0=ap._blocks(obs)["green0"]; d=ap.dir; R=grasp_R(np.arctan2(d[1],d[0]))
    pg=np.array([g0[0]-d[0]*0.02,g0[1]-d[1]*0.02,g0[2]+dz])
    tool=world_fk(q,base)[0]
    t0=max(np.dot(pg-tool,d),0.02)
    reached=None
    for t in np.arange(t0,-0.001,-0.01):
        p=pg-d*max(t,0)
        sols=ik_solutions(p,R,base,q,seeds=3,iters=100)
        if not sols: break
        obs,rej=servo(env,ap,obs,sols[0],8)
        q=ap._robot(obs)["q"]
        if rej: break
        reached=t
    tool=world_fk(q,base)[0]
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,_,_,_,_=env.step(a)
    hold=ap._robot(obs)['holding']
    print(f"dz={dz}: reached extra={reached} tool={np.round(tool,3)} GRASP={hold}")
    if hold:
        # try to lower to 0.83 while holding
        q=ap._robot(obs)["q"]
        for z in [tool[2]-0.02,tool[2]-0.04]:
            p=np.array([tool[0],tool[1],z])
            sols=ik_solutions(p,R,base,q,seeds=3,iters=100)
            if not sols: print("   lower ikfail"); break
            obs,rej=servo(env,ap,obs,sols[0],6); q=ap._robot(obs)["q"]
            print("   lower to",round(z,3),"rej",rej,"hold",ap._robot(obs)['holding'],"g0",np.round(ap._blocks(obs)['green0'],3))
            if rej: break
env.close()
