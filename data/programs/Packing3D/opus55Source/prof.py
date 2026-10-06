import cProfile, pstats, sys
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.packing3d:Packing3DEnv",count_kwarg="num_parts",count_object_prefix="part",design_counts=[1,2,3,4],eval_counts=[1,2,3,4])
ap=GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=0, options={'object_count':int(sys.argv[1])})
ap._twin(int(sys.argv[1]))
cProfile.run("ap.reset(s,info)","prof.out")
p=pstats.Stats("prof.out"); p.sort_stats("cumulative").print_stats(35)
