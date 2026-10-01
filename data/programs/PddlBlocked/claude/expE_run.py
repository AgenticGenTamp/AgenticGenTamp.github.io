import sys, numpy as np
from env_client import make_env
from farroute import FarRoute

seeds = [int(s) for s in (sys.argv[1:] or ["0","1","2","3"])]
env = make_env()
tot = {}
for seed in seeds:
    obs, info = env.reset(seed=seed)
    pol = FarRoute()
    pl = pol.reset(obs, info)
    print("seed", seed, "n_spares", info["object_count"], "pick", pl["name"],
          "base", np.round(pl["base"],3), "dir", np.round(pl["dirv"],2))
    steps = 0; term = False; R = 0.0
    log = []
    for t in range(1000):
        a = pol.get_action(obs)
        obs, rew, term, trunc, info = env.step(a)
        steps += 1; R += rew
        log.append((t, pol.phase, pol.stall, pol.mode))
        if term or trunc:
            break
    r = obs.data[obs.get_object_from_name("robot")]
    print("  -> steps", steps, "term", term, "trunc", trunc, "phase", pol.phase,
          "stalls", sum(1 for l in log if l[2]>0), "base", np.round(r[:3],3))
    ph = []
    for l in log:
        if not ph or ph[-1][0] != l[1]: ph.append([l[1], l[0]])
    print("  phases", ph)
    tot[seed] = (steps, term)
env.close()
print("SUMMARY", tot)
