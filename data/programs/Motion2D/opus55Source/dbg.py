import time,sys
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
import approach
c,seed=int(sys.argv[1]),int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.motion2d:Motion2DEnv", count_kwarg="num_passages", count_object_prefix="obstacle", bilevel_env_name="motion2d", design_counts=[c], eval_counts=[c])
s,info=env.reset(seed=seed, options={'object_count':c})
a=GeneratedApproach(env.action_space, env.observation_space, {})
orig=a._plan; cnt=[0]
def p(st,th): cnt[0]+=1; return orig(st,th)
a._plan=p
t0=time.time(); a.reset(s,info); print("reset t",time.time()-t0, a.path is None)
ta=0;te=0
for k in range(1000):
    t=time.time(); act=a.get_action(s); ta+=time.time()-t
    t=time.time(); s,r,term,_,_=env.step(act); te+=time.time()-t
    if term: break
print(k,term,cnt,ta,te,a.total_stuck)
