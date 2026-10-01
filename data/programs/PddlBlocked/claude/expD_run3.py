import numpy as np, math, sys, time
from expD_lib import *; from expD_core import *; from expD_plan import *
env = make_env(); t0=time.time()
seeds=[int(x) for x in sys.argv[1].split(",")]
for seed in seeds:
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info); qh = ap._robot(obs)["q"].copy()
    d = np.array(ap.dir); B = ap._blocks(obs)
    print(f"### seed {seed} dir={np.round(d[:2],3)} yaw={math.atan2(d[1],d[0]):+.3f} blk={np.round(B['blocker'],3)} g0={np.round(B['green0'],3)}")
    for tgt in ["blocker","green0"]:
        blk = B[tgt]
        if tgt=="blocker" and blk[2] < 0.5:
            print(f"  {tgt}: already removed by builtin"); continue
        cands = candidates(blk, d)
        print(f"  {tgt}: {len(cands)} candidate base poses (in-bounds & table-clear>=0.36)")
        print(f"    {'back':>5}{'lat':>7}{'dyaw':>6} {'bx':>6}{'by':>6}{'clr':>6} {'why':>10} {'pre':>4}{'gap':>7} {'stop':>6}{'gr':>3}{'st':>4}")
        got=False
        for (bk,la,dy,bt,c) in cands[:10]:
            obs2, info2 = env.reset(seed=seed)
            ap2 = GeneratedApproach(env.action_space, env.observation_space, {}); ap2.reset(obs2, info2)
            obs2,o = approach_grasp(env,ap2,obs2,blk,d,bt,qh,mode="direct")
            gp = 99 if o['gap'] is None else o['gap']
            print(f"    {bk:5.2f}{la:7.3f}{dy:6.1f} {bt[0]:6.2f}{bt[1]:6.2f}{c:6.2f} {o['base_why']:>10} {str(o['pre_ok'])[:1]:>4}{gp:7.3f} {o['stop']:>6}{str(o['grasped'])[:1]:>3}{o['steps']:4d}")
            sys.stdout.flush()
            if o['grasped']: got=True; break
        print(f"  {tgt}: SUCCESS={got}")
print("t=%.1f"%(time.time()-t0))
