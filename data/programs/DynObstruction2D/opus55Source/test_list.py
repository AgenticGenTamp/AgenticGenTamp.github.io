import sys, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
pairs = [tuple(map(int, p.split(":"))) for p in sys.argv[1].split(",")]
counts = sorted({p[0] for p in pairs})
env = VariableObjectCountEnv(
    constant_object_env_path="kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction",
    design_counts=counts, eval_counts=counts)
ok_n = 0
for n, seed in pairs:
    state, info = env.reset(seed=seed, options={"object_count": n})
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(state, info)
    ok = False
    t0 = time.time()
    for t in range(300 + 150 * n):
        state, _, term, _, _ = env.step(ap.get_action(state))
        if term:
            ok = True
            break
    ok_n += ok
    print(n, seed, ok, t + 1, ap.phase, f"{time.time()-t0:.1f}s", flush=True)
print("success", ok_n, len(pairs))
