import logging, numpy as np, math
logging.disable(logging.WARNING)
import approach as A
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
    count_kwarg="num_cubes", count_object_prefix="cube", constant_object_env_kwargs={'num_boxes': 1},
    design_counts=[1], eval_counts=[1])
for seed in [0,4]:
    s,_ = env.reset(seed=seed, options={'object_count':1})
    sim = A.Sim(1); sim.load(s)
    off, thg = sim.pick('box0'); 
    he = np.array([0.1,0.15,0.1]); ex,ey = (he[1],he[0]) if abs(math.sin(thg))>0.5 else (he[0],he[1])
    c = np.array([0.8-ex, -0.4+ey, 0.5]); sim.try_place('box0', c, off, thg)
    bp = sim.obj_pos('box0'); print("box", np.round(bp,3), ex, ey)
    off, thg = sim.pick('cube0'); cp0 = sim.checkpoint()
    zb = bp[2]-0.1+0.01+0.025
    for gy in [0, math.pi/2]:
        for dx in [-ex+0.04, -ex/2, 0, ex/2, ex-0.04]:
            for dy in [-ey+0.04, 0, ey-0.04]:
                sim.restore(cp0)
                try:
                    sim.try_place('cube0', np.array([bp[0]+dx,bp[1]+dy,zb]), off, thg, pre_h=0.25, gy=gy); r="OK"
                except A.PlanFail as e: r=str(e)
                print(f"gy={gy:.2f} dx={dx:.2f} dy={dy:.2f} {r}")
