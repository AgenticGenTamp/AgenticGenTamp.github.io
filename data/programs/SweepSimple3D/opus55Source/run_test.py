import sys, time, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SweepSimple3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube_", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[1,5,10,50], eval_counts=[1,5,10,50])
def run(seed, n, max_steps=None, verbose=False):
    s, info = env.reset(seed=seed, options={'object_count': n})
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(s, info)
    max_steps = max_steps or env.max_steps_for_count(n)
    t0 = time.time()
    for t in range(max_steps):
        a = ap.get_action(s)
        s, r, term, trunc, _ = env.step(a)
        if verbose and t % int(verbose) == 0:
            b, arm, g, cubes, w = ap._read(s)
            print(t, 'base', b.round(3), ap.queue[0].name if ap.queue else '-', 'grip', round(g,2), 'cubes', [c.round(3).tolist() for c in cubes][:3])
        if term: break
    b, arm, g, cubes, w = ap._read(s)
    inside = sum((0.075<=c[0]<=0.925) and (0.95<=c[1]<=1.425) for c in cubes)
    print(f'seed {seed} n {n} term {term} steps {t+1} inside {inside}/{len(cubes)} time {time.time()-t0:.1f}', flush=True)
    return term, t+1
if __name__ == '__main__':
    n = int(sys.argv[1]); seeds = [int(x) for x in sys.argv[2].split(',')]
    ms = int(sys.argv[3]) if len(sys.argv)>3 else None
    vb = int(sys.argv[4]) if len(sys.argv)>4 else False
    for sd in seeds: run(sd, n, ms, vb)
