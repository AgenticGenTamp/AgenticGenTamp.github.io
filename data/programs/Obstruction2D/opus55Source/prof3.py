import sys, time, cProfile, pstats
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import approach
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
c, seed, vi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
s,info=env.reset(seed=seed, options={'object_count':c})
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap._pick_cache={}
m,target,surface,obst=ap._parse(s)
cProfile.run("r=ap._plan_mode(m,target,surface,obst,time.time()+30,*approach.VARIANTS[vi])","prof.out")
pstats.Stats("prof.out").sort_stats("cumulative").print_stats(18)
