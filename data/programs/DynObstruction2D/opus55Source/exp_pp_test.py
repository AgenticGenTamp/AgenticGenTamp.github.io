import sys
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from exp_pickplace import PickPlace, _pp_parse

def make(n):
    return VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
        count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[n], eval_counts=[n])

def free_x(state, name, rng):
    _, b, others = _pp_parse(state, name)
    w = b["r"] - b["l"]
    cands = []
    for x in np.arange(0.5, 2.75, 0.01):
        if abs(x - b["x"]) < 0.3: continue
        if all(x + w/2 + 0.25 < o["l"] or x - w/2 - 0.25 > o["r"] for o in others):
            cands.append(x)
    return float(rng.choice(cands)) if cands else None

verbose = "-v" in sys.argv
scramble = "-s" in sys.argv
res = []
envs = {n: make(n) for n in range(4)}
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][0] != "-" else 40):
    for n in range(4):
        env = envs[n]
        state, _ = env.reset(seed=seed, options={"object_count": n})
        if scramble:   # random robot motions -> arbitrary start pose
            srng = np.random.default_rng(1000 + seed)
            for _ in range(30):
                state, *_ = env.step(srng.uniform(env.action_space.low, env.action_space.high) * 0.999)
        names = [nm for nm in state.get_object_names() if nm == "target_block" or nm.startswith("obstruction")]
        rng = np.random.default_rng(seed * 10 + n)
        for nm in names:
            _, b, _ = _pp_parse(state, nm)
            if b["r"] - b["l"] >= 0.25 or abs(np.sin(b["theta"])) > 0.05 or b["bot"] > 0.11: continue
            if nm != "target_block" and rng.random() < 0.5: continue
            tx = free_x(state, nm, rng)
            if tx is None: continue
            r0, _, oth0 = _pp_parse(state, nm)
            feas = PickPlace.feasible(state, nm); pl = PickPlace.plan_grasp(state, nm)
            pp = PickPlace(env.action_space, nm, tx)
            t = 0
            while t < 500:
                a = pp.step(state)
                if a is None: break
                state, *_ = env.step(a); t += 1
                if verbose: print(t, pp.phase, np.round(a, 3))
            oth1 = _pp_parse(state, nm)[2]
            dist = max([np.hypot(o0["x"] - o1["x"], o0["y"] - o1["y"]) for o0, o1 in zip(oth0, oth1)] + [0.0])
            if dist > 0.01: print("NEIGHBOUR DISTURBED", seed, n, nm, round(dist, 3))
            res.append((seed, n, nm, tuple(round(r0[k], 2) for k in ('x', 'y', 'theta', 'arm_joint')), feas, pp.success, t, pp.info, round(tx,3), round(b["r"]-b["l"],3), round(b["top"]-b["bot"],3), pl and round(pl[2],4)))
            print(res[-1], flush=True)
            break  # one object per instance
ok = [r for r in res if r[5]]
fe = [r for r in res if r[4]]
print(f"total {len(res)} feasible {len(fe)} success {len(ok)}; success|feasible {sum(r[5] for r in fe)}/{len(fe)}; mean steps(success) {np.mean([r[6] for r in ok]) if ok else 0:.1f} max {max([r[6] for r in ok]) if ok else 0}")
