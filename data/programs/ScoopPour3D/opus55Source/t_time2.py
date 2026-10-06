import time, numpy as np, sys
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
N=int(sys.argv[1])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:ScoopPour3DEnv",count_kwarg="num_objects",count_object_prefix="cube_",constant_object_env_kwargs={'scene_bg': False},design_counts=[N],eval_counts=[N])
t=time.time(); s,info=env.reset(seed=0, options={'object_count':N}); print('reset',time.time()-t)
for k in range(3):
    t=time.time()
    for i in range(10): env.step(np.zeros(11,dtype=np.float32))
    print(N,'step', (time.time()-t)/10, flush=True)
oc=env._current_backend._object_centric_env
m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
print('ncon',d.ncon, 'solver', m.opt.solver, m.opt.iterations, m.opt.cone, m.opt.noslip_iterations)
print(env._base_steps, env._steps_per_object)
