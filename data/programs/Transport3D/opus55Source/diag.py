import sys, logging, numpy as np, math
logging.disable(logging.WARNING)
import os; os.environ['NOGATE']='1'
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach, PlanFail
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
    count_kwarg="num_cubes", count_object_prefix="cube", constant_object_env_kwargs={'num_boxes': 1},
    design_counts=[1,2,3], eval_counts=[1,2,3,4])
s, info = env.reset(seed=28, options={'object_count': 6})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.plan = ap._make_plan(s)
sim = ap.sims[6]
print("base", sim.base(), "ee", sim.ee())
cp = sim.checkpoint()
e = sim.ee(); yaw=0.0
for z in (0.5,0.55,0.6,0.65,0.7,0.75,0.8):
    sim.restore(cp)
    ok = sim.move_ee(np.array([e[0],e[1],z]), sim.base()[2]); print("lift to", z, ok, np.round(sim.ee(),3))
for z in (None, 0.6, 0.7, 0.8):
    sim.restore(cp)
    if z: sim.move_ee(np.array([e[0],e[1],z]), sim.base()[2])
    ok = sim.to_carry(); print("to_carry after lift", z, ok, np.round(sim.ee(),3), np.round(sim.joints(),2))
    if ok:
        b=sim.base()
        print("  back base", sim.move_base([b[0]-0.1, b[1]+0.1, b[2]]), " rot", sim.move_base([b[0], b[1], 0.5]))
from approach import PRE_H
for nm in ('cube0','cube1'):
  for phi in (0.0, 0.39, -0.39, 1.57, -1.57, 3.14):
    sim.restore(cp)
    g, gyaw, bg = sim.pick_geom(nm, phi, 0.5, 0.0, 0.0)
    try:
        sim.navigate(bg[0], bg[1], bg[2], final=(g+[0,0,PRE_H], gyaw), phase="slow"); r="ok"
        r += " down=%s" % sim.move_ee(g, gyaw)
    except PlanFail as ex: r=str(ex)
    print(nm, phi, np.round(bg,2), r)
