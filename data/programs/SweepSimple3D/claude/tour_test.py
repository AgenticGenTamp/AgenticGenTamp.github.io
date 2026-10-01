import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]) if len(sys.argv)>1 else 1
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap.goal_override=[np.array(p) for p in [(1.75,0.10),(0.85,0.10),(0.80,0.85),(2.15,0.85),(2.15,1.7)]]
ap.reset(obs,info)
names=[n for n in obs.get_object_names() if n.startswith("cube_")]
def cs(o): return {n:(round(float(o.get(o.get_object_from_name(n),"x")),3),round(float(o.get(o.get_object_from_name(n),"y")),3)) for n in names}
last=None; tot=0
for t in range(1000):
    a=ap.get_action(obs)
    obs,rew,term,trunc,info=env.step(a); tot+=rew
    if last is None or abs(rew-last)>1e-9:
        print("REW",t,rew,cs(obs),flush=True); last=rew
    if t%100==0: print("t",t,"gi",ap.gi,"ph",ap.phase,cs(obs),flush=True)
    if term or trunc: print("TERM",t); break
print("return",tot,cs(obs))
env.close()
