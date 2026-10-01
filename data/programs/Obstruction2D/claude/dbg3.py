import sys, numpy as np
from env_client import make_env
import approach as A
seed=int(sys.argv[1]); oc=int(sys.argv[2]) if len(sys.argv)>2 else None
env = make_env()
ap = A.GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=seed) if oc is None else env.reset(seed=seed, options={'object_count':oc})
ap.reset(obs, info)
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: round(float(obs.get(o,f)),4) for f in obs.type_features[o.type]}
for n in sorted(obs.get_object_names()):
    r=d(obs,n)
    if 'arm_joint' in r: print(n, round(r['x'],3), round(r['y'],3))
    else: print(n, "x=[%.3f,%.3f]"%(r['x'],r['x']+r['width']), "y=%.3f top=%.3f"%(r['y'],r['y']+r['height']))
last=None
for t in range(400):
    a = ap.get_action(obs)
    cur=(ap.task and ap.task['obj'], ap.wp_i)
    if cur!=last:
        r=d(obs,'robot')
        print(t, "task",cur, "wp",ap.waypoints[ap.wp_i] if ap.wp_i<len(ap.waypoints) else None, "robot",round(r['x'],3),round(r['y'],3),r['vacuum'], "moves",ap.moves)
        last=cur
    obs, rew, term, trunc, info = env.step(np.asarray(a,dtype=np.float32))
    if term: print("TERM",t); break
env.close()
