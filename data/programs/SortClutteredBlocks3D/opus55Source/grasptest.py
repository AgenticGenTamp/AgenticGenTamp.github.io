import sys, numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import approach as A
yaw=float(sys.argv[1]); dz=float(sys.argv[2]); ox=float(sys.argv[3]) if len(sys.argv)>3 else 0
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[4], eval_counts=[4])
s,info=env.reset(seed=0, options={'object_count': 4})
oc = env._current_backend._object_centric_env
sim = oc._robot_env.sim; m=sim.model.mj_model; d=sim.data.mj_data
# move cubes 2-4 away
for k,c in enumerate(['cube2','cube3','cube4']):
    j=m.body(c).jntadr[0]; a=m.jnt_qposadr[j]; d.qpos[a:a+3]=[0.2,-0.25+0.05*k,0.42]
j=m.body('cube1').jntadr[0]; a=m.jnt_qposadr[j]; d.qpos[a:a+7]=[0,0,0.41,1,0,0,0]
mujoco.mj_forward(m,d)
ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
s,*_=env.step(np.zeros(11)); ap.reset(s,{})
base=A.BASE_TARGET
def run(pos, n, grip, ramp=None):
    global s
    ap.grip=grip; ap._set_arm_target(base,pos,yaw,ramp=ramp)
    for i in range(n):
        s,*_=env.step(ap._act(*ap._robot(s)[:2], ap._vel))
        b,arm,_=ap._robot(s); ee=ap._ee(b,arm); c=ap._pose(s,'cube1')
        print(' ee',ee.round(3),'cube',c[:3].round(3), 'grip', grip)
run([ox,0,0.6],8,0.62)
run([ox,0,0.41+dz+0.06],5,0.62)
run([ox,0,0.41+dz],4,0.62)
print('close')
run([ox,0,0.41+dz],3,1.0)
print('lift')
run([ox,0,0.6],5,1.0,ramp=[0.1,0.3,0.6,1.0])
for i in range(m.ngeom):
    b=m.geom_bodyid[i]; bn=m.body(b).name
    if any(k in bn.lower() for k in ['pad','finger','follower','driver','spring']) :
        print(m.geom(i).name, bn, d.geom_xpos[i].round(3), m.geom_contype[i], m.geom_conaffinity[i])
