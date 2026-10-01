import numpy as np, math
from expD_lib import *

def grasp_pose(blk, d):
    return np.array([blk[0]-0.02*d[0], blk[1]-0.02*d[1], blk[2]+0.03])

def approach_grasp(env, ap, obs, blk, d, base_t, qh, mode="direct",
                   pre_back=0.22, zhigh=1.02, step=0.02, verbose=False):
    """Returns dict with results."""
    out = dict(base_ok=False, base_why="", pre_ok=False, pre_err=None,
               gap=None, grasped=False, steps=0, ikfail=False, stop="full", lat_err=None)
    obs, ok, why, berr = drive(env, ap, obs, base_t, qh)
    out["base_why"] = why; out["base_err"] = berr
    if not ok: return obs, out
    out["base_ok"] = True
    gp = grasp_pose(blk, d); R = grasp_R(math.atan2(d[1], d[0]))
    pre = gp - d*pre_back
    r = ap._robot(obs); base = r["base"]; q = r["q"]
    ns = 0
    if mode == "high":
        hi = pre.copy(); hi[2] = zhigh
        qh1 = try_ik(hi, R, base, q, seeds=6)
        if qh1 is None: out["ikfail"] = True; return obs, out
        obs, rej = servo(env, ap, obs, base_t, qh1, n=20); ns += 20
        if rej: out["base_why"] = "high_rej"; return obs, out
        q = ap._robot(obs)["q"]
        # descend
        for z in np.arange(zhigh-0.04, pre[2]-1e-9, -0.04):
            p = pre.copy(); p[2] = z
            qn = try_ik(p, R, base, q, seeds=3)
            if qn is None: continue
            obs, rej = servo(env, ap, obs, base_t, qn, n=8); ns += 8
            if rej: out["base_why"] = "descend_rej"; return obs, out
            q = ap._robot(obs)["q"]
    qp = try_ik(pre, R, base, q, seeds=6)
    if qp is None: out["ikfail"] = True; return obs, out
    obs, rej = servo(env, ap, obs, base_t, qp, n=20); ns += 20
    q = ap._robot(obs)["q"]
    tp = world_fk(q, base)[0]
    out["pre_err"] = float(np.linalg.norm(tp - pre))
    out["pre_ok"] = (not rej) and out["pre_err"] < 0.02
    if not out["pre_ok"]:
        out["gap"] = float(np.dot(gp - tp, d)); out["steps"] = ns; return obs, out
    # cartesian step in
    n_in = int(round(pre_back/step))
    for k in range(1, n_in+1):
        p = pre + d*(step*k)
        qn = try_ik(p, R, base, q, seeds=3)
        if qn is None: out["stop"]="ik"; break
        obs, rej = servo(env, ap, obs, base_t, qn, n=6); ns += 6
        q = ap._robot(obs)["q"]
        tp2 = world_fk(q, base)[0]
        if rej: out["stop"]="rej"; break
        if np.linalg.norm(tp2 - p) > 0.03: out["stop"]="track"; break
    tp = world_fk(ap._robot(obs)["q"], base)[0]
    out["gap"] = float(np.dot(gp - tp, d))
    out["lat_err"] = float(np.linalg.norm((gp-tp) - np.dot(gp-tp, d)*d))
    a = np.zeros(11, dtype=np.float32); a[10] = -1.0
    obs,_,_,_,_ = env.step(a); ns += 1
    out["grasped"] = bool(ap._robot(obs)["holding"])
    out["steps"] = ns
    return obs, out
