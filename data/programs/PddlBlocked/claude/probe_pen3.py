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
for high_z, final_dz in [(0.98,0.03),(1.02,0.03),(0.95,0.05),(0.98,0.0)]:
    obs,info=env.reset(seed=1)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    for t in range(19):
        a=ap.get_action(obs); obs,_,_,_,_=env.step(a)
    r=ap._robot(obs); base=r["base"]; q=r["q"]
    g0=ap._blocks(obs)["green0"]; d=ap.dir; R=grasp_R(np.arctan2(d[1],d[0]))
    pg=np.array([g0[0]-d[0]*0.02,g0[1]-d[1]*0.02,g0[2]])
    # go up, forward above pen, then descend
    pts=[]
    tool=world_fk(q,base)[0]
    pts.append(np.array([tool[0],tool[1],high_z]))
    for f in [0.14,0.07,0.0]:
        pts.append(np.array([pg[0]-d[0]*f,pg[1]-d[1]*f,high_z]))
    for z in np.arange(high_z-0.03, g0[2]+final_dz-0.001, -0.03):
        pts.append(np.array([pg[0],pg[1],z]))
    pts.append(np.array([pg[0],pg[1],g0[2]+final_dz]))
    stop=None; nst=0
    for p in pts:
        sols=ik_solutions(p,R,base,q,seeds=3,iters=100)
        if not sols: stop=("ikfail",p); break
        obs,rej=servo(env,ap,obs,sols[0],8)
        q=ap._robot(obs)["q"]
        if rej: stop=("rej",np.round(world_fk(q,base)[0],3)); break
    if stop is None:
        a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,_,_,_,_=env.step(a)
        print(f"high={high_z} dz={final_dz}: GRASP={ap._robot(obs)['holding']} tool={np.round(world_fk(ap._robot(obs)['q'],base)[0],3)}")
    else:
        print(f"high={high_z} dz={final_dz}: {stop}")
env.close()
