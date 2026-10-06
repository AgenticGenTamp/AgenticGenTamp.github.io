import numpy as np, approach
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ap = approach.GeneratedApproach(env.action_space, env.observation_space, {})
orig = ap.move
def move(base_t, q_t, grip, **kw):
    n=0
    for a in orig(base_t, q_t, grip, **kw):
        n+=1; yield a
    print("move", n, end=" | ")
ap.move = move
obs,info = env.reset(seed=0); ap.reset(obs,info)
for t in range(520):
    obs,*_ = env.step(ap.get_action(obs))
