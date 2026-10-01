import sys
import numpy as np
from env_client import make_env
import approach as A
sd=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=sd)
ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info)
printed=False
for i in range(env.max_steps):
    a=ap.get_action(obs)
    if ap.phase=='push' and not printed:
        printed=True
        hk_obs=A.Hook(obs[9:11],obs[11],obs[18],obs[19],obs[17])
        hk_mod=ap._hook_for_robot(obs,obs[:2],obs[2])
        print("obs hook C",np.round(obs[9:12],4))
        print("mod hook C",np.round(hk_mod.C,4), round(hk_mod.t,4))
        M=obs[20:22]
        for k,bar in enumerate(hk_obs.bars()):
            print("  bar",k,"distM",round(A.dist_pt_rect(M,*bar),4))
        print("plan mode",ap.plan['mode'],"v",np.round(ap.plan['v'],3),"need",round(ap.plan['need'],3))
        print("robot",np.round(obs[:3],3),"pre",np.round(ap.pre_pose,3),"final",np.round(ap.final_pos,3))
        print("planC",np.round(ap.plan['C'],3),"t",round(ap.plan['t_hook'],3))
    obs,r,term,trunc,info=env.step(a)
    if ap.phase=='push':
        M=obs[20:22]
        hk=A.Hook(obs[9:11],obs[11],obs[18],obs[19],obs[17])
        print(f"  push t={i} robot={np.round(obs[:2],3)} M={np.round(M,4)} db0={A.dist_pt_rect(M,*hk.bars()[0]):.4f} db1={A.dist_pt_rect(M,*hk.bars()[1]):.4f} dT={np.linalg.norm(M-obs[29:31]):.4f}")
    if term or trunc: print("DONE",i,term); break
env.close()
