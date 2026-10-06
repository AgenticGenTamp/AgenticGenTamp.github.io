import sys, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
c, seed = int(sys.argv[1]), int(sys.argv[2])
s,info=env.reset(seed=seed, options={'object_count':c})
ap=GeneratedApproach(env.action_space, env.observation_space, {})
m,target,surface,obst=ap._parse(s)
print("robot", m.rx, m.ry, "surface", surface)
for n,b in sorted(m.blocks.items(), key=lambda kv: kv[1][0]): print(n, [round(v,3) for v in b])
t=time.time(); ap.reset(s,info); print("plan len", len(ap.plan), time.time()-t)
n='obstruction1'
print("spots", ap._free_spots(m, n, [(surface[0]-0.005, surface[0]+surface[2]+0.005)]))
b=m.blocks[n]
for gx in ap._grasp_xs(b):
    r = ap._travel(m, gx, b[1]+b[3]+0.2+0.0125, 0.2, 0.0)
    print(gx, r is not None)
