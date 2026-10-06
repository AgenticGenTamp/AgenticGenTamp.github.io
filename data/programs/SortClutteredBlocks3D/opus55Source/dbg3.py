import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach, _yaw_from_quat
n=int(sys.argv[1]); sd=int(sys.argv[2]); T=int(sys.argv[3])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[n], eval_counts=[n])
ap=GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=sd, options={'object_count': n}); ap.reset(s,info)
for i in range(T):
    a=ap.get_action(s)
    s,r,te,tr,_=env.step(a)
    if ap.phase in ('descend','close','lift'):
        b,arm,g=ap._robot(s); ee=ap._ee(b,arm); _,R=ap.kin.fk(b,arm)
        c=ap._pose(s,ap.cur)
        print(i, ap.phase, ap.cur, 'ee',ee.round(3),'cube',c[:3].round(3),'eeyaw',np.degrees(np.arctan2(R[1,0],R[0,0])).round(1),'cmdyaw',np.degrees(ap.yaw).round(1),'cubeyaw',np.degrees(_yaw_from_quat(*c[3:])).round(1), 'Rz', R[:,2].round(2), 'g', a[10])
