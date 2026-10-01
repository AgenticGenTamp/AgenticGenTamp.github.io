import sys, numpy as np
from env_client import make_env
from farroute import FarRoute, robot_state, _cart_path, _arm_cost, GRASP_BACK, GRASP_UP
from approach import world_fk, world_chain, ik_solutions, grasp_R, dq_wrap

seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
pol=FarRoute(); pl=pol.reset(obs,info)
d3=np.array([pl["dirv"][0],pl["dirv"][1],0.0])
tgt=pl["block"]-d3*GRASP_BACK+np.array([0,0,GRASP_UP])
print("blk",np.round(pl["block"],3),"tgt",np.round(tgt,3),"base",np.round(pl["base"],3))
for t in range(60):
    a=pol.get_action(obs); obs,rew,term,trunc,info=env.step(a)
    if pol.phase in ("approach","close") and pol.stall>=2: break
r=robot_state(obs); p,R=world_fk(r["q"],r["base"])
print("stall step",t,"phase",pol.phase,"tool",np.round(p,3),"err",np.round(tgt-p,3))
pts,_=world_chain(r["q"],r["base"]); print("chain",np.round(np.array(pts),2).tolist())

def try_q(obs,qt,tag):
    ok=True
    for _ in range(6):
        r=robot_state(obs); dq=np.clip(dq_wrap(qt-r["q"]),-0.2,0.2)
        if np.max(np.abs(dq))<1e-3: break
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        o2,*_=env.step(a)
        if np.allclose(robot_state(o2)["q"],r["q"],atol=1e-6): ok=False; obs=o2; break
        obs=o2
    p,_=world_fk(robot_state(obs)["q"],robot_state(obs)["base"])
    print("  ",tag,"ok",ok,"tool",np.round(p,3),"derr %.3f"%np.linalg.norm(tgt-p))
    return obs,ok

# A: alternative IK branches at the exact target
sols=ik_solutions(tgt,pl["R"],r["base"],r["q"],seeds=14,iters=140,rng=np.random.default_rng(0))
sols.sort(key=lambda s: _arm_cost(s,r["base"])[0])
print("n_sols",len(sols))
for i,s in enumerate(sols[:4]):
    obs,ok=try_q(obs,s,"branch%d cost %.2f"%(i,_arm_cost(s,r["base"])[0]))
    if ok:
        a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,*_=env.step(a)
        print("     grasp_active",robot_state(obs)["holding"])
        if robot_state(obs)["holding"]: break
        a[10]=1.0; obs,*_=env.step(a)
env.close()
