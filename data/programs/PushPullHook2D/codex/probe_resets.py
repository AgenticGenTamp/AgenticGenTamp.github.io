import math
import numpy as np
from env_client import make_env


def main():
    rows = []
    env = make_env()
    for seed in range(500):
        obs, info = env.reset(seed=seed)
        r = obs[0:2]
        h = obs[9:11]
        b = obs[20:22]
        t = obs[29:31]
        rows.append((seed, *r, obs[2], obs[4], *h, obs[11], *b, *t,
                     np.linalg.norm(r-b), np.linalg.norm(b-t), np.linalg.norm(h-b)))
    env.close()
    a = np.array([x[1:] for x in rows])
    names = ['rx','ry','rtheta','arm','hx','hy','htheta','bx','by','tx','ty','r-b','b-t','h-b']
    for i, name in enumerate(names):
        vals = a[:, i]
        print(f'{name:7s} min={vals.min(): .4f} max={vals.max(): .4f} mean={vals.mean(): .4f} unique={len(np.unique(np.round(vals,4)))}')
    print('fixed metadata seed0')
    env = make_env(); o,_=env.reset(seed=0); env.close()
    print(o.tolist())
    print('closest button-target')
    for x in sorted(rows, key=lambda x:x[-2])[:12]: print(x)
    print('farthest button-target')
    for x in sorted(rows, key=lambda x:-x[-2])[:12]: print(x)
    print('closest robot-button')
    for x in sorted(rows, key=lambda x:x[-3])[:12]: print(x)


if __name__ == '__main__':
    main()
