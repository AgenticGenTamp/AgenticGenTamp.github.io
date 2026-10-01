import numpy as np
from env_client import make_env
from approach import GeneratedApproach, Robot, tool_R, yaw_of, quat_R
import fk
env=make_env()
for seed,tilt,ang in [(2,0.0,0.0),(2,0.0,np.pi/2),(2,0.5,0.0),(19,0.0,0.0),(19,0.5,0.0)]:
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    R=Robot(obs); name=sorted(R.blocks)[0]; b=R.blocks[name].copy()
    Rw=tool_R(ang,tilt,-np.pi/2)
    gp=np.array(b[:3])-0.05*Rw[:,0]
    pre=gp-0.15*Rw[:,0]
    plan=None
    for cand in sorted(ap.base_candidates(), key=lambda c: ap.base_cost(R.base,c)):
        if ap.ik_at(cand,gp,Rw) is not None and ap.ik_at(cand,pre,Rw) is not None:
            plan=cand;break
    if plan is None: print(seed,tilt,ang,"unreachable"); continue
    for wp in ap.base_path(R.base,plan):
        for i in range(20):
            d=np.array(wp)-R.base
            if np.abs(d).max()<1e-4: break
            a=np.zeros(11); a[:3]=np.clip(d,-0.2,0.2)
            obs,*_=env.step(np.float32(a)); R=Robot(obs)
    okall=True
    for tgt in [pre,gp]:
        for i in range(40):
            dq,conv=ap.cart_step(R,tgt,Rw,step_len=0.05,ptol=1e-3,rtol=3e-3)
            if conv: break
            if dq is None: okall=False;break
            prev=R.q.copy(); obs,*_=env.step(ap.act_arm(dq)); R=Robot(obs)
            if np.allclose(prev,R.q): okall=False;break
        else: okall=False
    p,Rtool,_=R.tool()
    obs,*_=env.step(np.float32([0]*10+[-1.0])); R2=Robot(obs)
    b2=R2.blocks[name]
    print(f"seed{seed} tilt{tilt} ang{ang} conv{okall} held={R2.holding}")
    print("   before q",np.round(b[3:7],3),"after q",np.round(b2[3:7],3))
    print("   tool R x-axis",np.round(Rtool[:,0],3),"y",np.round(Rtool[:,1],3))
env.close()
