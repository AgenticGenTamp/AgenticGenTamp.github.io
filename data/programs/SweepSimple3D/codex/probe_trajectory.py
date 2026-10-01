import sys
import numpy as np
from env_client import make_env


def get(state, name, features):
    o = state.get_object_from_name(name)
    return np.array([float(state.get(o, f)) for f in features])


def run(seed, actions, label):
    env = make_env()
    s, info = env.reset(seed=seed)
    cubes = sorted(n for n in s.get_object_names() if n.startswith("cube_"))
    names = ["robot", "wiper_0"] + cubes
    def poses(st):
        return {n: get(st, n, ["pos_base_x", "pos_base_y", "pos_base_rot"])
                if n == "robot" else get(st, n, ["x", "y", "z"]) for n in names}
    p0 = poses(s)
    print("RUN", label, "n", len(cubes), "initial", {n: np.round(v,3).tolist() for n,v in p0.items()})
    for i, a in enumerate(actions):
        s, rew, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        if i % 2 == 0 or rew != -1.0 or term or trunc or i == len(actions)-1:
            pp = poses(s)
            moved = {n: np.round(pp[n]-p0[n],3).tolist() for n in names
                     if np.linalg.norm(pp[n]-p0[n]) > .003}
            print(i+1, "r", rew, "moved", moved, "done", term, trunc)
        if term or trunc:
            break
    env.close()


def act(x=0, y=0, yaw=0, grip=0):
    a = np.zeros(11); a[0:3] = [x,y,yaw]; a[10] = grip; return a


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv)>1 else 0
    run(seed, [act(y=-.1)]*20, "global_y_negative")
    run(seed, [act(x=-.1)]*20, "global_x_negative")
    run(seed, [act(y=-.1)]*9 + [act(y=.1)]*9, "toward_wiper_then_back")
    run(seed, [act(grip=1)]*4 + [act(y=.1, grip=1)]*8, "close_then_back")
    run(seed, [act(x=.1)]*3 + [act(y=-.1)]*22, "offset_right_then_down")
    run(seed, [act(x=-.1)]*3 + [act(y=-.1)]*22, "offset_left_then_down")
