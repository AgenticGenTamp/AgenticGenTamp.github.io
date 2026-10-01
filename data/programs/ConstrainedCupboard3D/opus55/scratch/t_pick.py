import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
env = make_env()
for seed in [0,2]:
    obs, info = env.reset(seed=seed); S = Sim(env, obs)
    for name in sorted(S.rods()):
        ok = pick_rod(S, name)
        print(seed, name, ok, np.round(S.rods()[name][:3],3), np.round(grasp_point_world(S),3), len(S.rew))
        move_ee_world(S, grasp_point_world(S)*[1,1,0]+[0,0,0.1], Rdown(quat_yaw(S.rods()[name])), grip=0.0, steps=40)
        S.goto(grip=0.0, steps=5)
env.close()
