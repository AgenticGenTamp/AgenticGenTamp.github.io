import sys, numpy as np
from env_client import make_env
from farroute import FarRoute, robot_state, _cart_path, GRASP_BACK, GRASP_UP
from approach import world_fk, ik_solutions, grasp_R, dq_wrap
seed=int(sys.argv[1]); env=make_env(); obs,info=env.reset(seed=seed)
pol=FarRoute(); pl=pol.reset(obs,info)
d3=np.array([pl["dirv"][0],pl["dirv"][1],0.0])
tgt=pl["block"]-d3*GRASP_BACK+np.array([0,0,GRASP_UP])
for t in range(60):
    a=pol.get_action(obs); obs,*_=env.step(a)
    if pol.phase in ("approach","close") and pol.stall>=2: break
base=robot_state(obs)["base"]
def goto(obs,pt,tag):
    q0=robot_state(obs)["q"]
    pth=_cart_path(base,q0,[pt],pl["R"],seeds=8,iters=120)
    if not pth: print("  ",tag,"IKfail"); return obs,False
    ok=True
    for _ in range(6):
        r=robot_state(obs); dq=np.clip(dq_wrap(pth[0]-r["q"]),-0.2,0.2)
        if np.max(np.abs(dq))<1e-3: break
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq; o2,*_=env.step(a)
        if np.allclose(robot_state(o2)["q"],r["q"],atol=1e-6): ok=False; obs=o2; break
        obs=o2
    p,_=world_fk(robot_state(obs)["q"],base)
    print("  ",tag,"ok",ok,"tool",np.round(p,3))
    return obs,ok
print("tgt",np.round(tgt,3),"grip",robot_state(obs)["grip"])
obs,_=goto(obs,tgt+np.array([0,0,0.08]),"above tgt+8cm")
obs,_=goto(obs,tgt+np.array([0,0.30,0.0]),"side y+0.30 same x")
obs,_=goto(obs,tgt+np.array([0,0.30,0.0])+np.array([-0.06,0,0]),"side deeper x-0.06")
# open gripper fully then retry straight-in
a=np.zeros(11,dtype=np.float32); a[10]=1.0; obs,*_=env.step(a)
print("   grip after open",robot_state(obs)["grip"])
obs,_=goto(obs,tgt+np.array([0.06,0,0.0]),"back to preline")
obs,ok=goto(obs,tgt,"straight in (open)")
a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,*_=env.step(a)
print("   holding",robot_state(obs)["holding"])
env.close()
