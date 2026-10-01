import sys, numpy as np
from env_client import make_env
import approach as A
seed=int(sys.argv[1])
env = make_env()
ap = A.GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=seed)
ap.reset(obs, info)
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: round(float(obs.get(o,f)),4) for f in obs.type_features[o.type]}
for n in sorted(obs.get_object_names()): print(n, d(obs,n))
prev=None
for t in range(120):
    a = ap.get_action(obs)
    r=d(obs,'robot')
    print(t, "task",ap.task and ap.task['obj'],"wp",ap.wp_i, ap.waypoints[ap.wp_i] if ap.wp_i<len(ap.waypoints) else None,
          "robot", round(r['x'],3), round(r['y'],3), round(r['arm_joint'],3), r['vacuum'], "act", np.round(a,3))
    obs, rew, term, trunc, info = env.step(np.asarray(a,dtype=np.float32))
    if term: print("TERM at",t); break
    if t%5==0: print("   objs", {n:(round(d(obs,n)["x"],3),round(d(obs,n)["y"],3)) for n in sorted(obs.get_object_names()) if n!="robot"})
env.close()
