import time, numpy as np
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
import approach as A
k=1; env=E([k],[k]); obs,info=env.reset(seed=0,options={'object_count':k})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
t=time.time(); ap.reset(obs,info); print('reset',time.time()-t)
flat,_=A.obs_to_flat(obs)
blocks=A.block_info(flat,k)
print(blocks)
t=time.time()
s=ap.sim; P=ap.planner
qc=flat[:10]
b=blocks[0]; byaw=A.quat_yaw(b['quat'])
for m in range(4):
    t=time.time()
    q=P.solve_ik(b['pos']+np.array([0,0,0.03]), A.top_quat(A.wrap(byaw+m*np.pi/2)), qc)
    print(m, time.time()-t, None if q is None else (np.round(q,2), A.nsteps(qc,q)))
for t_ in range(30):
    t=time.time()
    a=ap.get_action(obs)
    obs,r,term,_,_=env.step(a)
    f,_=A.obs_to_flat(obs)
    print(t_, round(time.time()-t,3), np.round(a,2), 'grasp',f[11], 'term',term, len(ap.plan))
    if term: break
