import sys, numpy as np
from env_client import make_env

def feats(obs, name):
    o = obs.get_object_from_name(name)
    return dict(zip(obs.type_features[o.type], [float(v) for v in obs.data[o]]))

def run(seed=0, count=1, push_to=2.5, log_every=20):
    env = make_env()
    obs, info = env.reset(seed=seed, options={'object_count': count})
    rb = feats(obs, 'robot')
    cub = feats(obs, 'cuboid_0')
    print(f"seed={seed} count={count} info={info}")
    print(f"  robot base ({rb['pos_base_x']:.3f},{rb['pos_base_y']:.3f},{rb['pos_base_rot']:.3f})")
    print(f"  cuboid_0 ({cub['x']:.3f},{cub['y']:.3f},{cub['z']:.3f})")
    rewards = []
    tgt_y = cub['y']
    step = 0
    # Phase 1: back off in x to behind the cuboid while aligning y
    for phase, (tx, ty) in enumerate([(cub['x'] - 0.55, tgt_y), (push_to, tgt_y)]):
        for _ in range(400):
            rb = feats(obs, 'robot')
            a = np.zeros(11, dtype=np.float32)
            a[0] = np.clip(2.0 * (tx - rb['pos_base_x']), -0.1, 0.1)
            a[1] = np.clip(2.0 * (ty - rb['pos_base_y']), -0.1, 0.1)
            a[2] = np.clip(-2.0 * rb['pos_base_rot'], -0.1, 0.1)
            obs, r, term, trunc, info = env.step(a)
            step += 1
            rewards.append(r)
            c = feats(obs, 'cuboid_0')
            if abs(r - (-1.0)) > 1e-9:
                print(f"  !! step {step} r={r} cuboid=({c['x']:.3f},{c['y']:.3f},{c['z']:.3f}) base=({rb['pos_base_x']:.3f},{rb['pos_base_y']:.3f})")
            if step % log_every == 0:
                print(f"  step {step} r={r} phase{phase} base=({rb['pos_base_x']:.3f},{rb['pos_base_y']:.3f}) cub=({c['x']:.3f},{c['y']:.3f},{c['z']:.3f}) term={term} trunc={trunc}")
            if term or trunc:
                print(f"  EPISODE END step={step} term={term} trunc={trunc} r={r} info={info}")
                env.close(); return
            if abs(tx - rb['pos_base_x']) < 0.01 and abs(ty - rb['pos_base_y']) < 0.01:
                break
    print("  unique rewards:", sorted(set(round(x,6) for x in rewards)))
    env.close()

if __name__ == '__main__':
    run(seed=int(sys.argv[1]) if len(sys.argv)>1 else 0,
        count=int(sys.argv[2]) if len(sys.argv)>2 else 1)
