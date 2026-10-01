from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
print("r0", pose(obs,0), "r1", pose(obs,1))
# rover0 theta=pi. push +x
obs,m,d = try_move(env,obs,0.2,0.0,0.0,0); print("r0 dx=+0.2 ->", np.round(d,5))
obs,m,d = try_move(env,obs,0.0,0.2,0.0,0); print("r0 dy=+0.2 ->", np.round(d,5))
# rover1 theta=0
obs,m,d = try_move(env,obs,0.2,0.0,0.0,1); print("r1 dx=+0.2 ->", np.round(d,5))
obs,m,d = try_move(env,obs,0.0,0.2,0.0,1); print("r1 dy=+0.2 ->", np.round(d,5))
# rotate rover1 to pi/2 then translate
for _ in range(10):
    obs,m,d = try_move(env,obs,0.0,0.0,0.4,1)
print("r1 pose after rot", np.round(pose(obs,1),5))
obs,m,d = try_move(env,obs,0.1,0.0,0.0,1); print("r1(th=1.57?) dx=+0.1 ->", np.round(d,5))
obs,m,d = try_move(env,obs,0.0,0.1,0.0,1); print("r1 dy=+0.1 ->", np.round(d,5))
# sub-maximal
obs,m,d = try_move(env,obs,0.05,0.03,0.0,1); print("r1 (0.05,0.03) ->", np.round(d,6))
obs,m,d = try_move(env,obs,0.01,0.0,0.0,1); print("r1 (0.01,0) ->", np.round(d,6))
obs,m,d = try_move(env,obs,0.001,0.0,0.0,1); print("r1 (0.001,0) ->", np.round(d,6))
obs,m,d = try_move(env,obs,0.2,0.2,0.0,1); print("r1 (0.2,0.2) diag ->", np.round(d,6))
# out of range action
obs,m,d = try_move(env,obs,0.5,0.0,0.0,1); print("r1 (0.5,0) ->", np.round(d,6))
# dtheta range
obs,m,d = try_move(env,obs,0.0,0.0,0.9,1); print("r1 dth=0.9 ->", np.round(d,6))
print("theta now", pose(obs,1)[2])
for _ in range(20):
    obs,m,d = try_move(env,obs,0.0,0.0,0.4,1)
print("theta wrap check", pose(obs,1)[2])
env.close()
