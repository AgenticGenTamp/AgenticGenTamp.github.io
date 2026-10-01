import numpy as np, math, sys, time
from expD_lib import *; from expD_core import *; from expD_plan import *
from expD_blk import remove_blocker2
env = make_env(); t0=time.time()

def g0_active(obs):
    return float(obs.get(obs.get_object_from_name("green0"), "grasp_active"))

def prep(env, seed):
    """blocker removed, ready state"""
    ap, obs, n = remove_blocker(env, seed, max_steps=80)
    if ap._blocks(obs)["blocker"][2] < 0.5: return ap, obs, n, "builtin"
    ap, obs, n2, ok, why = remove_blocker2(env, seed)
    return ap, obs, n2, f"custom({ok},{why})"

for seed in [int(x) for x in sys.argv[1].split(",")]:
    ap, obs, npre, tag = prep(env, seed)
    d = np.array(ap.dir); g0 = ap._blocks(obs)["green0"]
    print(f"### seed {seed} dir={np.round(d[:2],3)} yaw={math.atan2(d[1],d[0]):+.3f} g0={np.round(g0,3)} blockerRemoval={tag} steps={npre}")
    cands = candidates(g0, d, backs=(0.90,0.85,0.95,0.80,0.75,0.70), lats=(-0.188,0.0,0.188,-0.30,0.30,-0.40), dyaws=(0.0,0.3,-0.3), clear=0.34)
    print(f"  {len(cands)} feasible base poses. {'back':>5}{'lat':>7}{'dyaw':>6} {'bx':>6}{'by':>6}{'clr':>6} {'why':>10}{'pre':>4}{'gap':>7} {'stop':>6}{'G0':>3}{'st':>4}")
    qh = home_q(env, seed); nwin=0
    for (bk,la,dy,bt,c) in cands[:24]:
        ap2, obs2, _, _ = prep(env, seed)
        obs2,o = approach_grasp(env,ap2,obs2,g0,d,bt,qh,mode="direct")
        ga = g0_active(obs2); gp = 99 if o['gap'] is None else o['gap']
        print(f"    {bk:5.2f}{la:7.3f}{dy:6.1f} {bt[0]:6.2f}{bt[1]:6.2f}{c:6.2f} {o['base_why']:>10}{str(o['pre_ok'])[:1]:>4}{gp:7.3f} {o['stop']:>6}{int(ga):3d}{o['steps']:4d}")
        sys.stdout.flush()
        if ga > 0.5:
            nwin += 1
            if nwin >= 3: break
    print(f"  seed {seed}: green0 grasped in {nwin} of the tried poses")
print("t=%.1f"%(time.time()-t0))
