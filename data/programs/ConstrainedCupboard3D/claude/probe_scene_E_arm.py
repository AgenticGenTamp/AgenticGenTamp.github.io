import sys, numpy as np
from env_client import make_env

def feats(obs, name):
    o = obs.get_object_from_name(name)
    return dict(zip(obs.type_features[o.type], [float(v) for v in obs.data[o]]))

def main(seed=0, count=1, nsteps=400, mode='flail'):
    rng = np.random.default_rng(0)
    env = make_env()
    obs, info = env.reset(seed=seed, options={'object_count': count})
    c0 = feats(obs, 'cuboid_0')
    print("start cuboid", {k: round(v,3) for k,v in c0.items() if k in 'xyz'}, "info", info)
    tx, ty = c0['x'] - 0.35, c0['y']
    step = 0; rews = []
    prev = np.array([c0['x'], c0['y'], c0['z']])
    for i in range(nsteps):
        rb = feats(obs, 'robot')
        a = np.zeros(11, dtype=np.float32)
        if i < 120:
            a[0] = np.clip(2.0*(tx-rb['pos_base_x']), -0.1, 0.1)
            a[1] = np.clip(2.0*(ty-rb['pos_base_y']), -0.1, 0.1)
            a[2] = np.clip(-2.0*rb['pos_base_rot'], -0.1, 0.1)
        else:
            a[3:10] = rng.uniform(-0.1, 0.1, 7)
            a[10] = 1.0 if (i//40) % 2 else 0.0
        obs, r, term, trunc, info = env.step(a); step += 1; rews.append(r)
        c = feats(obs, 'cuboid_0')
        cur = np.array([c['x'], c['y'], c['z']])
        if np.linalg.norm(cur-prev) > 0.02:
            print(f" step {step} MOVED cub=({c['x']:.3f},{c['y']:.3f},{c['z']:.3f}) r={r}")
            prev = cur
        if abs(r+1.0) > 1e-9:
            print(f" step {step} RDIFF r={r} cub=({c['x']:.3f},{c['y']:.3f},{c['z']:.3f})")
        if term or trunc:
            print(f" END step={step} term={term} trunc={trunc} r={r}"); break
    print("unique rewards", sorted(set(round(x,6) for x in rews)))
    cend = feats(obs, 'cuboid_0')
    print("end cuboid", {k: round(v,3) for k,v in cend.items()})
    env.close()

main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
