"""Custom blocker removal for seeds where the built-in policy gets stuck."""
import numpy as np, math
from expD_lib import *
from expD_core import grasp_pose, approach_grasp

def remove_blocker2(env, seed, back=0.85, lat=-0.188, verbose=False):
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    qh = ap._robot(obs)["q"].copy()
    b = ap._blocks(obs); d = np.array(ap.dir); blk = b["blocker"]
    ns = 0
    if blk[2] < 0.5:   # already on floor
        return ap, obs, 0, True, "already"
    from expD_plan import candidates
    ok = False
    for (bk,la,dy,bt,c) in candidates(blk, d)[:10]:
        obs, o = approach_grasp(env, ap, obs, blk, d, bt, qh, mode="direct")
        ns += o["steps"]
        if o["grasped"] and abs(o["gap"]) < 0.03: ok = True; break
        if o["grasped"]:
            a=np.zeros(11,dtype=np.float32); a[10]=1.0; obs,_,_,_,_=env.step(a); ns+=1
    if not ok:
        return ap, obs, ns, False, "grasp_fail"
    # retreat along -dir at grasp height, then lift, then drop far away
    R = grasp_R(math.atan2(d[1], d[0])); gp = grasp_pose(blk, d)
    q = ap._robot(obs)["q"]; base = ap._robot(obs)["base"]
    gp = grasp_pose(blk, d)
    for k in range(1, 12):
        p = gp - d*(0.03*k); p[2] = gp[2] + min(0.15, 0.02*k)
        qn = try_ik(p, R, base, q, seeds=3)
        if qn is None: break
        obs, rej = servo(env, ap, obs, bt, qn, n=6); ns += 6
        if rej: break
        q = ap._robot(obs)["q"]
    a = np.zeros(11, dtype=np.float32); a[10] = 1.0
    obs,_,_,_,_ = env.step(a); ns += 1
    bnow = ap._robot(obs)["base"]
    obs, rej = servo(env, ap, obs, bnow, qh, n=30); ns += 30
    if rej: ns += 0
    blk2 = ap._blocks(obs)["blocker"]
    return ap, obs, ns, (blk2[2] < 0.5 or np.linalg.norm(blk2[:2]-ap._blocks(obs)["green0"][:2]) > 0.4), f"z={blk2[2]:.2f}"
