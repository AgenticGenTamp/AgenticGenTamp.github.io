"""Focused multiseed probe of drawer preload feedback; never imported by policy."""
import sys
import numpy as np
from env_client import make_env


mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
seeds = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else list(range(8))
for seed in seeds:
    env = make_env()
    obs, _ = env.reset(seed=seed)
    initial = obs.copy()

    def act(goal):
        global obs
        a = np.zeros(11, np.float32)
        a[:3] = np.clip(0.8 * (np.asarray(goal) - obs[125:128]), -0.1, 0.1)
        a[10] = 1.0
        obs, reward, term, trunc, info = env.step(a)
        return term or trunc

    def go(goal, count):
        for _ in range(count):
            if act(goal):
                break

    phase0_yaw = 3.10 if mode in ("normalize", "normalize_pose") else initial[127]
    phase0_x = 1.24 if mode == "normalize_pose" else initial[125]
    go([phase0_x, -1.35, phase0_yaw], 20)
    go([0.78, -1.35, 1.74], 28)
    go([0.78, -0.92, 1.74], 15)
    after_route = obs.copy()
    max_open = float(np.max(obs[103:109]))
    latched = None
    trace = []
    for t in range(57):
        opened = float(np.max(obs[103:109]))
        max_open = max(max_open, opened)
        if mode in ("baseline", "normalize", "normalize_pose"):
            if latched is None and opened > 0.52:
                latched = obs[125:128].copy()
            goal = latched if latched is not None else [0.94, -0.92, 1.74]
        elif mode == "ratchet":
            # Push +x until extension stops improving, then preserve contact pose.
            if latched is None and opened > 0.54:
                latched = obs[125:128].copy()
            goal = latched if latched is not None else [1.02, -0.92, 1.74]
        elif mode == "feedback":
            # Advance gently with opening progress; stop before chassis rebounds.
            gx = min(1.02, 0.84 + 0.22 * max(0.0, opened) / 0.55)
            if latched is None and opened > 0.50:
                latched = obs[125:128].copy()
            goal = latched if latched is not None else [gx, -0.92, 1.74]
        elif mode == "turnpush":
            # If contact jams yaw below target, command extra turn while pushing.
            gyaw = 2.05 if obs[127] < 1.72 and opened < 0.45 else 1.74
            if latched is None and opened > 0.52:
                latched = obs[125:128].copy()
            goal = latched if latched is not None else [0.98, -0.92, gyaw]
        elif mode == "retry":
            # Failed first contact: disengage south, pre-rotate, and recollide.
            if t < 10:
                goal = [0.94, -0.92, 1.74]
            elif max_open < 0.45 and t < 25:
                goal = [0.80, -1.55, 2.00]
            elif max_open < 0.45 and t < 42:
                goal = [0.80, -0.92, 2.00]
            else:
                goal = [1.05, -0.92, 2.00]
        else:
            raise ValueError(mode)
        act(goal)
        if t in (0, 9, 19, 29, 39, 56):
            trace.append((t, round(opened, 3), round(float(obs[125]), 3)))
    print(seed, "init", np.round(initial[125:128], 2), "route",
          round(float(np.max(after_route[103:109])), 3),
          "route_pose", np.round(after_route[125:128], 2),
          "final", round(float(np.max(obs[103:109])), 3),
          "max", round(max_open, 3), "pose", np.round(obs[125:128], 2),
          "trace", trace, flush=True)
    env.close()
