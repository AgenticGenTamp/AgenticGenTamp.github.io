import numpy as np
from env_client import make_env
from approach import GeneratedApproach, world_fk, ik_solutions, grasp_R, dq_wrap
def servo(env,ap,obs,base_t,qt,n=12):
    for k in range(n):
        r=ap._robot(obs); dq=dq_wrap(qt-r["q"]); db=np.array(base_t)-r["base"]
        db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
        if max(np.max(np.abs(dq)),np.max(np.abs(db)))<1e-7: return obs,False
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(dq,-0.2,0.2); a[:3]=np.clip(db,-0.2,0.2)
        prev=np.concatenate([r["base"],r["q"]])
        obs,_,_,_,_=env.step(a)
        r2=ap._robot(obs)
        if np.max(np.abs(np.concatenate([r2["base"],r2["q"]])-prev))<1e-9: return obs,True
    return obs,False
env=make_env()
obs,info=env.reset(seed=1)
ap0=GeneratedApproach(env.action_space,env.observation_space,{}); ap0.reset(obs,info)
g0=ap0._blocks(obs)["green0"]; d=ap0.dir
perp=np.array([-d[1],d[0],0.0])
cands=[]
for lat in [0.0,0.19,-0.19,0.35,-0.35]:
    for back in [0.7,0.55,0.85]:
        p=g0[:2]-d[:2]*back+perp[:2]*lat
        yaw=np.arctan2(g0[1]-p[1],g0[0]-p[0])
        cands.append((np.array([p[0],p[1],yaw]),lat,back))
for base,lat,back in cands:
    obs,info=env.reset(seed=1)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    for t in range(19):
        a=ap.get_action(obs); obs,_,_,_,_=env.step(a)   # removes blocker
    r=ap._robot(obs); q=r["q"]
    R=grasp_R(np.arctan2(d[1],d[0]))
    pg=np.array([g0[0]-d[0]*0.02,g0[1]-d[1]*0.02,g0[2]+0.03])
    p0=pg-d*0.22
    sols=ik_solutions(p0,R,base,q,seeds=6,iters=110)
    if not sols:
        print(f"lat={lat} back={back}: pregrasp IK fail"); continue
    obs,rej=servo(env,ap,obs,base,sols[0],25)
    if rej:
        print(f"lat={lat} back={back}: rejected reaching pregrasp"); continue
    q=ap._robot(obs)["q"]
    reached=None
    for t in np.arange(0.20,-0.001,-0.01):
        p=pg-d*max(t,0)
        s=ik_solutions(p,R,base,q,seeds=3,iters=100)
        if not s: break
        obs,rj=servo(env,ap,obs,base,s[0],8); q=ap._robot(obs)["q"]
        if rj: break
        reached=t
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,_,_,_,_=env.step(a)
    print(f"lat={lat} back={back}: stop_extra={reached} tool={np.round(world_fk(q,base)[0],3)} GRASP={ap._robot(obs)['holding']}")
env.close()
