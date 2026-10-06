import sys, numpy as np, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach, _yaw_from_quat
n=int(sys.argv[1]); sd=int(sys.argv[2]); T=int(sys.argv[3])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False}, design_counts=[n], eval_counts=[n])
s,info=env.reset(seed=sd, options={'object_count': n})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s,info)
t0=time.time(); nf=0; ph0=None; cur0=None
for i in range(T):
    a=ap.get_action(s)
    f=sum(ap.fail.values())
    if f>nf: print(i,'  FAIL',cur0,'in',ph0, ap._pose(s,cur0)[:3].round(3)); nf=f
    if ap.phase=='approach' and ap.steps_in_phase<=1 and (ap.cur!=cur0 or ph0!='approach'):
        p=ap._pose(s,ap.cur); print(i,'PICK',ap.cur,p[:3].round(3),'cyaw',round(np.degrees(_yaw_from_quat(*p[3:7]))),'yaw',round(np.degrees(ap.yaw)),'off',ap.goff.round(3),'score',round(ap.last_cands[0][4],4),'ncand',len(ap.last_cands))
    if ap.phase=='release' and ph0!='release': print(i,'RELEASE',ap.cur)
    if ap.phase=='papp' and ph0!='papp': print(i,'PUSH',ap.cur,ap._pose(s,ap.cur)[:2].round(3),ap.push_start.round(3),ap.push_end.round(3))
    ph0=ap.phase; cur0=ap.cur
    s,r,te,tr,_=env.step(a)
    if te: print('DONE',i); break
print('time',time.time()-t0)
