import sys, numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach, compose
from kinder.envs.utils import state_2d_has_collision
env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
s=int(sys.argv[1]); N=int(sys.argv[2])
obs, info = env.reset(seed=s)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for n in range(N):
    a = ap.get_action(obs); obs,_,_,_,_ = env.step(a)
a = ap.get_action(obs)
print("action", a)
H = compose(obs[[0,1,2]], ap.T); print("hook", obs[9:12], "pred", H)
wx, wy = ap.hook.hull_world((np.array(H[0]),np.array(H[1]),np.array(H[2]))); print(wx, wy)
ke = env.unwrapped._kinder_env if hasattr(env.unwrapped,'_kinder_env') else env._kinder_env
oc = ke.unwrapped._object_centric_env
st = oc._current_state.copy()
oc.step(a)
print("moved?", oc._current_state.get([o for o in st if o.name=='robot'][0],'x'))
