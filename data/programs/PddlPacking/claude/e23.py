import numpy as np, fk, ctrl, lib
from env_client import make_env
from approach import GeneratedApproach, Robot, tool_R, yaw_of
env=make_env()
for seed in [0,2,7,13]:
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    R=Robot(obs)
    name=sorted(R.blocks)[0]; b=R.blocks[name]
    byaw=yaw_of(*b[3:7])
    for ang in [0.0, np.pi/2]:
        obs,info=env.reset(seed=seed); R=Robot(obs); ap.reset(obs,info)
        Rw=tool_R(ang)
        gp=np.array([b[0],b[1],b[2]+0.05])
        base=None
        for cand in sorted(ap.base_candidates(), key=lambda c: ap.base_cost(R.base,c)):
            if ap.ik_at(cand,gp,Rw) is not None: base=cand;break
        if base is None: print(seed,"no base");continue
        # move base
        for wp in ap.base_path(R.base, base):
            for i in range(20):
                d=np.array(wp)-R.base
                if np.abs(d).max()<1e-4: break
                a=np.zeros(11); a[:3]=np.clip(d,-0.2,0.2)
                obs,r,t,tr,inf=env.step(np.float32(a)); R=Robot(obs)
        # cart to pre then gp
        for tgt,st in [(np.array([b[0],b[1],0.95]),0.1),(gp,0.05)]:
            for i in range(40):
                dq,conv=ap.cart_step(R,tgt,Rw,step_len=st)
                if conv or dq is None: break
                obs,r,t,tr,inf=env.step(ap.act_arm(dq)); R=Robot(obs)
        obs,r,t,tr,inf=env.step(np.float32([0]*10+[-1.0])); R=Robot(obs)
        print(seed,name,"blockyaw",round(byaw,2),"ang",round(ang,2),"grasp",R.holding,
              "blkq",np.round(R.blocks[name][3:7],3) if R.holding else "")
env.close()
