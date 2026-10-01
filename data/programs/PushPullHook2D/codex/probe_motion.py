import numpy as np
from env_client import make_env


def run(seed, action, steps=40):
    e = make_env(); o, _ = e.reset(seed=seed)
    print('start', seed, 'robot', o[:7], 'hook', o[9:12], 'button', o[20:22], 'target', o[29:31])
    for i in range(steps):
        p = o.copy(); o, r, term, trunc, info = e.step(np.asarray(action, np.float32))
        if i % 5 == 4 or term or trunc:
            print(i+1, 'rob', np.round(o[[0,1,2,4,6]],3), 'hook', np.round(o[9:12],3), 'button',np.round(o[20:22],3),r,term,trunc)
        if term or trunc: break
    e.close()


if __name__ == '__main__':
    run(0, [0,.05,0,0,0])
