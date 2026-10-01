import sys, numpy as np, time
from env_client import make_env
from approach import GeneratedApproach
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def run(seed, verbose=False, approach_kw=None):
    env = make_env()
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    for k, v in (approach_kw or {}).items(): setattr(ap, k, v)
    ap.reset(obs, info)
    tot = 0; last = None
    for t in range(env.max_steps):
        a = ap.get_action(obs)
        obs, r, te, tr, info = env.step(a)
        tot += r
        if verbose and (ap.phase != last):
            last = ap.phase
            print(t, ap.phase, ap.obj, "r=%.3f" % r, "LB", obs[0:3], "S1", obs[54:57], "S2", obs[70:73], "sw q", obs[41:45])
        if te or tr: break
    env.close()
    return te, t, tot, obs
if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [0]
    for s in seeds:
        te, t, tot, obs = run(s, verbose=len(seeds) == 1)
        print("seed", s, "success", te, "steps", t, "ret %.2f" % tot)
