import sys, time, collections
import numpy as np
from env_client import make_env
from approach_fast import GeneratedApproach
import os, importlib
toss_fast = importlib.import_module(os.environ.get("MOD", "toss_fast"))
import approach_fast; approach_fast.toss = toss_fast
seeds = [int(s) for s in sys.argv[1].split(',')]
nobj = int(sys.argv[2])
env = make_env()
tot = collections.Counter(); succ = 0; steps = []
for seed in seeds:
    obs, info = env.reset(seed=seed, options={'object_count': nobj})
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    getattr(toss_fast, 'THROWS', []).clear()
    ap.reset(obs, info); cnt = collections.Counter(); seq = []
    tpol = 0; land = {}; binp = toss_fast.objpos(obs, 'bin_0')
    for t in range(env.max_steps):
        t1 = time.time(); a = ap.get_action(obs); tpol += time.time() - t1
        p = getattr(toss_fast, "PHASE", [""])[0]; cnt[p] += 1
        if not seq or seq[-1][0] != p: seq.append([p, 0])
        seq[-1][1] += 1
        obs, r, term, trunc, info = env.step(a)
        for n in toss_fast.cube_names(obs):
            p = toss_fast.objpos(obs, n)
            v = toss_fast.objvel(obs, n)
            if n not in land and p[2] > 0.2 and v[2] < -1 and p[0] > binp[0] - 1.2:
                tt = (v[2] + np.sqrt(v[2]**2 + 2*9.81*(p[2]-0.025))) / 9.81
                land[n] = (round(p[0]+v[0]*tt-binp[0], 3), round(p[1]+v[1]*tt-binp[1], 3))
        if term or trunc: break
    for th in getattr(toss_fast, 'THROWS', []): print('  THROW', th[0], 'off', th[1].tolist(), 'w', th[2], 'qerr', th[3], 'v', th[4], 'land', land.get(th[0]))
    succ += term; steps.append(t + 1); tot.update(cnt)
    print(f"seed {seed} ok={term} steps={t+1} pol={tpol:.1f}s land={list(land.values())}", ' '.join(f"{p}:{n}" for p, n in seq), flush=True)
print(f"SUCCESS {succ}/{len(seeds)} mean_steps={np.mean(steps):.1f}")
print('avg per phase:', {k: round(v / len(seeds), 1) for k, v in sorted(tot.items())})
env.close()
