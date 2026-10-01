"""Helper: run approach.py's first two tasks to remove the blocker, then hand back."""
import numpy as np
from approach import GeneratedApproach, world_fk, ik_solutions, grasp_R, dq_wrap

def remove_blocker(env, seed, max_steps=60):
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    n = 0
    for t in range(max_steps):
        if ap.task_i >= 2 and not ap.queue:
            break
        a = ap.get_action(obs)
        obs, _, term, trunc, _ = env.step(a)
        n += 1
        if term or trunc:
            break
        if ap.task_i >= 2:
            break
    return ap, obs, n

def servo(env, ap, obs, base_t, qt, n=15):
    """Servo base+arm; returns (obs, rejected)."""
    for k in range(n):
        r = ap._robot(obs)
        dq = dq_wrap(qt - r["q"]); db = np.array(base_t) - r["base"]
        db[2] = (db[2] + np.pi) % (2 * np.pi) - np.pi
        if max(np.max(np.abs(dq)), np.max(np.abs(db))) < 1e-7:
            return obs, False
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = np.clip(dq, -0.2, 0.2); a[:3] = np.clip(db, -0.2, 0.2)
        prev = np.concatenate([r["base"], r["q"]])
        obs, _, _, _, _ = env.step(a)
        r2 = ap._robot(obs)
        if np.max(np.abs(np.concatenate([r2["base"], r2["q"]]) - prev)) < 1e-9:
            return obs, True
    return obs, False
