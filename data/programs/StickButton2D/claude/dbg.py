import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
s=int(sys.argv[1]) if len(sys.argv)>1 else 4
obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info)
def rf(o,n,f): return round(float(o.get(o.get_object_from_name(n),f)),3)
print("plan",ap.plan,"need",ap.need_stick,"grasp_pos",ap.grasp_pos)
for n in sorted(obs.get_object_names()):
    if n.startswith("button"): print(n, rf(obs,n,"x"), rf(obs,n,"y"))
print("stick",rf(obs,"stick","x"),rf(obs,"stick","y"))
for i in range(200):
    a=ap.get_action(obs)
    obs,r,term,trunc,inf=env.step(a)
    if i%5==0 or i<20:
        print(i,"act",np.round(a,3),"rob",rf(obs,"robot","x"),rf(obs,"robot","y"),rf(obs,"robot","theta"),rf(obs,"robot","arm_joint"),
              "stick",rf(obs,"stick","x"),rf(obs,"stick","y"),rf(obs,"stick","theta"),"carry",ap.carrying,"gp",ap.grasp_phase,"stall",ap.stall,"task",ap.plan[0] if ap.plan else None)
    if term: print("TERM",i);break
env.close()
