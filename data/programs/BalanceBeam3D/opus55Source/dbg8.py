import sys, numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
sd=int(sys.argv[1]); a0=int(sys.argv[2]); a1=int(sys.argv[3])
env = KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=sd); ap.reset(obs, info)
for t in range(a1):
    a = ap.get_action(obs)
    obs, r, term, trunc, info = env.step(a)
    if t>=a0:
        p,_ = ap.kin.fk(obs[16:19], obs[19:26])
        print(t, "b0",np.round(obs[0:3],3),"b1",np.round(obs[54:57],3),"ssq",np.round(obs[41:45],3),"site",np.round(p,3),"g",round(float(obs[26]),2))
