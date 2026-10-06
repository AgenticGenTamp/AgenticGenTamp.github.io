import sys, time, numpy as np
sys.path.insert(0,'.')
from approach import *
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[1], eval_counts=[1])
st,_=env.reset(seed=0)
ap=GeneratedApproach(None,None,{}); sim=ap._get_sim(1); flat=sim.sync(st,set_walls=True)
e=sim.env
with e.client():
    pl=Planner(sim,time.time()+60)
    t=time.time()
    g=pl.plan_grasp(flat, e.spares[0], max_keep=3, extra=(0,1,2,4))
    print("grasps", [x[0] for x in g], time.time()-t)
    if g:
        f1=g[0][3]
        e.set_state(f1); print("block", sim.ssp.get_pose(e.spares[0])[0])
        r=pl.plan_translate_to_plate(f1); print("translate", r is not None)
    import pybullet as p
    from pybullet_tools.utils import get_link_name
    e.set_state(f1); q0=f1[:10].astype(float)
    bp=np.array(sim.ssp.get_pose(e.spares[0])[0])
    q=q0.copy(); q[0]+=4.5-bp[0]; q[1]+=-0.3-bp[1]
    sim.set_conf(q); e._attachment.assign()
    print("base", np.round(q[:3],2), "coll", e._in_collision())
    for b in list(e.obstacles)+list(e.movables):
        for pt in p.getClosestPoints(e.robot, b, 0.0, physicsClientId=e.client_id):
            print(get_link_name(e.robot, pt[3]), b, round(pt[8],4))
        if b!=e.spares[0]:
          for pt in p.getClosestPoints(e.spares[0], b, 0.0, physicsClientId=e.client_id): print("held vs", b, round(pt[8],4))
