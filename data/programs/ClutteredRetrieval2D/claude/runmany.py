import sys, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
from approach import GeneratedApproach

def run(s):
    env = make_env()
    obs, info = env.reset(seed=s)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    t0=time.time()
    try:
        ap.reset(obs, info)
        steps=0; term=False
        for i in range(env.max_steps):
            a = ap.get_action(obs)
            obs, r, term, trunc, info2 = env.step(a)
            steps+=1
            if term or trunc: break
        res=(s, info.get("object_count"), steps, bool(term), round(time.time()-t0,1), ap.phase)
    except Exception as e:
        res=(s, info.get("object_count"), -1, False, round(time.time()-t0,1), "ERR:"+repr(e)[:120])
    env.close()
    return res

seeds=[int(x) for x in sys.argv[1:]]
with ThreadPoolExecutor(max_workers=8) as ex:
    out=list(ex.map(run, seeds))
for r in sorted(out): print(r)
print("solved", sum(1 for r in out if r[3]), "/", len(out))
print("mean steps solved", np.mean([r[2] for r in out if r[3]]).round(1))
print("fails", [r[0] for r in out if not r[3]])
