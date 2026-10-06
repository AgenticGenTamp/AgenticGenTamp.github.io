import sys, time
from multiprocessing import Pool
def run(args):
    c, seed = args
    from robocode.environments.variable_object_count_env import VariableObjectCountEnv
    from approach import GeneratedApproach
    env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
    s, info = env.reset(seed=seed, options={'object_count': c})
    t = time.time()
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(s, info)
    maxs = 300 + 150 * c
    for i in range(maxs):
        s, r, term, trunc, _ = env.step(ap.get_action(s))
        if term: return (c, seed, True, i + 1, time.time() - t)
        if time.time() - t > 60: break
    return (c, seed, False, i + 1, time.time() - t)
if __name__ == "__main__":
    counts = [int(x) for x in sys.argv[1].split(",")]
    a, b = [int(x) for x in sys.argv[2].split("-")]
    jobs = [(c, s) for c in counts for s in range(a, b)]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as p:
        res = p.map(run, jobs, chunksize=1)
    import collections
    by = collections.defaultdict(list)
    for r in res: by[r[0]].append(r)
    for c in counts:
        rs = by[c]
        print(f"count={c} success={sum(r[2] for r in rs)}/{len(rs)} mean_steps={sum(r[3] for r in rs)/len(rs):.1f} max_time={max(r[4] for r in rs):.1f}")
    print("FAILS", [(r[0], r[1]) for r in res if not r[2]])
    print("SLOWEST", sorted([(round(r[4],1), r[0], r[1]) for r in res])[-5:])
    print("TOTAL mean steps", sum(r[3] for r in res)/len(res))
