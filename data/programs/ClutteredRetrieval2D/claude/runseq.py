import sys, os, time, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seeds=[int(x) for x in sys.argv[1:]]
env=make_env()
out=[]
for s in seeds:
    OC=os.environ.get('OC')
    obs, info = env.reset(seed=s, options={'object_count': int(OC)}) if OC else env.reset(seed=s)
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
        import traceback; traceback.print_exc()
        res=(s, info.get("object_count"), -1, False, round(time.time()-t0,1), "ERR:"+repr(e)[:100])
    out.append(res); print(res, flush=True)
print("solved", sum(1 for r in out if r[3]), "/", len(out))
print("mean steps", round(float(np.mean([r[2] for r in out if r[3]])),1))
print("fails", [r[0] for r in out if not r[3]])
