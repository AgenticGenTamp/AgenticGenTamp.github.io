import sys, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
count=int(sys.argv[1]); seed=int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[count], eval_counts=[count])
t0=time.time(); s, info = env.reset(seed=seed, options={'object_count': count}); tr=time.time()-t0
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
tp=te=0
for t in range(300+150*count):
    t0=time.time(); a=ap.get_action(s); tp+=time.time()-t0
    t0=time.time(); s,rw,term,trunc,info=env.step(a); te+=time.time()-t0
    if term: break
print(f"steps={t+1} reset={tr:.2f} policy={tp:.2f} env={te:.2f}")
