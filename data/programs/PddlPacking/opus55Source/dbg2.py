import time, numpy as np, pybullet as p
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
import approach as A
k=1; env=E([k],[k]); obs,info=env.reset(seed=0,options={'object_count':k})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info)
flat,_=A.obs_to_flat(obs)
b=A.block_info(flat,k)[0]; byaw=A.quat_yaw(b['quat'])
s=ap.sim; P=ap.planner; qc=flat[:10]
q=P.solve_ik(b['pos']+np.array([0,0,0.03]), A.top_quat(byaw), qc)
s.set_state(flat)
acts=P.path_actions(qc,q,A.nsteps(qc,q),-1.0)
s.set_state(flat)
for a in acts:
    o=np.array(s.env.step(np.asarray(a,np.float32))[0])
    print(np.round(o[:10],3), o[11])
with s.env.client():
    from pybullet_tools import utils as U
    s.set_state(flat); s.set_conf(q)
    print('incoll', s.env._in_collision())
    for body in [s.env.table, s.env.plate]+list(s.env.blocks):
        print(body, U.pairwise_collision(s.robot, body))
    tool=U.get_link_pose(s.robot,s.tool)
    print('offset', U.multiply(U.invert(tool), U.get_pose(s.env.blocks[0]))[0])
    # which links collide with block
    for l in U.get_all_links(s.robot):
        if U.pairwise_link_collision(s.robot,l,s.env.blocks[0]): print('link',l,U.get_link_name(s.robot,l))
        if U.pairwise_link_collision(s.robot,l,s.env.table): print('tlink',l,U.get_link_name(s.robot,l))
