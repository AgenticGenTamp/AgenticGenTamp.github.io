import sys, numpy as np, time
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
import approach as A
k=5; seed=0; T=int(sys.argv[1])
env=E([k],[k]); obs,info=env.reset(seed=seed,options={'object_count':k})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(T):
    a=ap.get_action(obs); obs,r,term,_,_=env.step(a)
flat,_=A.obs_to_flat(obs)
blocks=A.block_info(flat,k)
todo=[i for i,b in enumerate(blocks) if not ap._block_done(b)]
print('todo',todo, 'q',np.round(flat[:10],2))
s=ap.sim; P=ap.planner; qc=flat[:10].copy()
for bi in todo:
    b=blocks[bi]; byaw=A.quat_yaw(b['quat'])
    for m in range(4):
        yaw=A.wrap(byaw+m*np.pi/2)
        task=A.TaskGrasp(b['pos'],yaw)
        r=A.slp_solve(s,qc,qc,task,A.base_region(qc))
        if r is None: print(bi,m,'slp fail'); continue
        q=r[0]; n=A.nsteps(qc,q)
        acts=P.path_actions(qc,q,n,-1.0)
        ok,fin,_=P.simulate(flat,acts)
        info=''
        if not ok:
            # find which step collides
            s.set_state(flat)
            for j,a in enumerate(acts):
                o=np.array(s.env.step(np.asarray(a,np.float32))[0])
                if np.abs(A.conf_delta(o[:10], (qc+(j+1)*(acts[0][:10])))).max()>1e-3:
                    info=f'collide at step {j}'; 
                    with s.env.client():
                        from pybullet_tools.utils import pairwise_collision, get_link_name, get_all_links, pairwise_link_collision
                        s.set_state(flat); s.set_conf(qc+(j+1)*acts[0][:10])
                        for body in [s.env.table, s.env.plate]+list(s.env.blocks):
                            if pairwise_collision(s.robot, body):
                                links=[get_link_name(s.robot,l) for l in get_all_links(s.robot) if pairwise_link_collision(s.robot,l,body)]
                                info+=f' body {body} links {links}'
                    break
        print(bi,m,'t=%.3f'%r[1],'n',n,'ok',ok, ok and fin[11]>0.5, info)
