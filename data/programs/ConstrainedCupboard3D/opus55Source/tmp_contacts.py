import sys, numpy as np, mujoco
from test_approach import make_env
import approach as ap
env=make_env()
n,seed,t0,t1=[int(x) for x in sys.argv[1:5]]
s,info=env.reset(seed=seed, options={'object_count':n})
re=env._current_backend._object_centric_env._robot_env
m,d=re.sim.model.mj_model, re.sim.data.mj_data
A=ap.GeneratedApproach(env.action_space, env.observation_space, {}); A.reset(s,info)
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
for t in range(t1):
    a=A.get_action(s)
    if t>=t0:
        rob=s.get_object_from_name('robot')
        c,u,_=A._rod(s,A.cur)
        cs=set()
        for i in range(d.ncon):
            g1,g2=d.contact[i].geom1,d.contact[i].geom2
            n1=m.geom(g1).name or m.body(m.geom_bodyid[g1]).name; n2=m.geom(g2).name or m.body(m.geom_bodyid[g2]).name
            if 'cuboid' in n1+n2 or 'cupboard' in n1+n2: cs.add(n1+'|'+n2)
        print(t,A.phase,'base',np.round([s.get(rob,'pos_base_x'),s.get(rob,'pos_base_y'),s.get(rob,'pos_base_rot')],3),'pinch',np.round(d.site_xpos[sid],3),'rod',np.round(c,3),np.round(u,2),sorted(cs))
    s,r,term,_,_=env.step(a)
