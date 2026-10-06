import numpy as np, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(
    constant_object_env_path="kinder.envs.dynamic3d.task_families:Dynamo3DEnv",
    count_kwarg="num_objects",
    count_object_prefix="obstacle_chair",
    constant_object_env_kwargs={'scene_bg': False},
    design_counts=[1,2,3], eval_counts=[1,2,3])
print("action_space", env.action_space)
print("low", env.action_space.low, "\nhigh", env.action_space.high)
for t in env.type_features: print("TYPE", t.name, env.type_features[t])
print("max_steps", [env.max_steps_for_count(c) for c in [1,2,3]])
for cnt in [3,1,2]:
    try:
        obs, info = env.reset(seed=0, options={"object_count": cnt})
    except Exception as e:
        print("RESET FAIL count", cnt, repr(e)[:300]); continue
    print("=== count", cnt, "current_count", env.current_count, "info", info)
    for o in obs: print(o.name, o.type.name, np.round(obs[o],3).tolist())
    if cnt != 3: continue
    robot = [o for o in obs if o.type.name.startswith("tidybot") or "robot" in o.name][0]
    v = obs[robot]
    feats = env.type_features[robot.type]
    print("robot feats", list(zip(feats, np.round(v,3))))
    t0=time.time()
    act = np.zeros(env.action_space.shape)
    for i in range(5):
        obs, r, term, trunc, info = env.step(act)
        print("step", i, "r", r, term, trunc, "robot", np.round(obs[robot][:12],3).tolist())
    print("time per step", (time.time()-t0)/5)
