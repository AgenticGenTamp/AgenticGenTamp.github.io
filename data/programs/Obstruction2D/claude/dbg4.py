import sys, numpy as np
from env_client import make_env
import approach as A
env = make_env(); ap = A.GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=52, options={'object_count':4})
ap.reset(obs, info)
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: round(float(obs.get(o,f)),4) for f in obs.type_features[o.type]}
for t in range(60):
    a = ap.get_action(obs)
    if t>=33:
        r=d(obs,'robot')
        print(t,"wp",ap.wp_i, "task",ap.task and ap.task['obj'],
              None if ap.wp_i>=len(ap.waypoints) else (round(ap.waypoints[ap.wp_i]['x'],3),round(ap.waypoints[ap.wp_i]['y'],3)),
              "robot",round(r['x'],3),round(r['y'],3),r['vacuum'],"act",np.round(a,3), "repl",ap.replans)
    obs, rew, term, trunc, info = env.step(np.asarray(a,dtype=np.float32))
    if term: print("TERM",t); break
print({n:(round(d(obs,n)['x'],3),round(d(obs,n)['y'],3)) for n in sorted(obs.get_object_names())})
env.close()
