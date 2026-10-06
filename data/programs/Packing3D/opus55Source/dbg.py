import sys, time
import approach; approach.DEBUG=True
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.packing3d:Packing3DEnv",count_kwarg="num_parts",count_object_prefix="part",design_counts=[1,2,3,4],eval_counts=[1,2,3,4])
ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=int(sys.argv[2]), options={'object_count':int(sys.argv[1])})
ap._twin(int(sys.argv[1]))
t=time.time(); ap.reset(s,info); print("plan len", len(ap._plan), "time", time.time()-t)
