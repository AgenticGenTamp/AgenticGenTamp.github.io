import numpy as np, json, math
from env_client import make_env
from approach import GeneratedApproach, world_fk, ik_solutions, grasp_R, dq_wrap
from helper_remove import remove_blocker, servo

def perp(d):  # left-hand perp: (-dy, dx)
    return np.array([-d[1], d[0], 0.0])

def base_for(g0, d, back, lat, dyaw=0.0):
    """base = green0_xy - dir*back + perp*lat ; lat<0 => to the right."""
    p = np.array(g0[:2]) - np.array(d[:2])*back + perp(d)[:2]*lat
    return np.array([p[0], p[1], math.atan2(d[1], d[0]) + dyaw])

def home_q(env, seed):
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    return ap._robot(obs)["q"].copy()

def drive(env, ap, obs, base_t, qh, nret=25, ndrive=60):
    """retract arm to qh, then drive base. returns (obs, ok, why, base_err)"""
    r = ap._robot(obs)
    obs, rej = servo(env, ap, obs, r["base"], qh, n=nret)
    note = "retract_rej|" if rej else ""
    obs, rej = servo(env, ap, obs, base_t, qh, n=ndrive)
    r = ap._robot(obs)
    e = np.array(base_t) - r["base"]; e[2] = (e[2]+np.pi)%(2*np.pi)-np.pi
    err = float(np.max(np.abs(e)))
    if rej: return obs, False, note+"drive_rej", err
    return obs, (err < 1e-3), note+("drive_slow" if err >= 1e-3 else "ok"), err

def tool(ap, obs):
    r = ap._robot(obs)
    return world_fk(r["q"], r["base"])[0]

def try_ik(p, R, base, q0, seeds=4, iters=110):
    s = ik_solutions(p, R, base, q0, seeds=seeds, iters=iters)
    if not s: return None
    # pick closest to q0
    return min(s, key=lambda q: np.max(np.abs(dq_wrap(q - q0))))
