"""Probe reuse of one grasped box to seat two cubes before releasing it."""
import numpy as np
from env_client import make_env
from probe_box_tool import Q, OFF, command, v, xyz


def step(e, s, a):
    s, _, term, trunc, _ = e.step(np.asarray(a, dtype=np.float32))
    return s, term or trunc


def seat(e, s, name, slot):
    """Reproduce approach.py's validated box-tool sequence."""
    c = xyz(s, name)
    u = np.asarray([slot[0] - c[0], slot[1] - c[1]])
    u /= max(np.linalg.norm(u), 1e-6)
    side = c[:2] - u * .18
    for _ in range(15):
        p = xyz(s, "box0")
        a = np.zeros(11); a[:2] = np.clip(side-p[:2], -.2, .2); a[10] = -1
        s, done = step(e, s, a)
    for _ in range(10):
        p = xyz(s, "box0")
        a = np.zeros(11); a[:2] = np.clip(np.asarray(slot)-p[:2], -.08, .08)
        a[4] = -.10; a[10] = -1
        s, done = step(e, s, a)
    for _ in range(35):
        c = xyz(s, name)
        a = np.zeros(11); a[:2] = np.clip(np.asarray(slot)-c[:2], -.10, .10); a[10] = -1
        s, done = step(e, s, a)
    for _ in range(12):
        c = xyz(s, name)
        a = np.zeros(11); a[:2] = np.clip(np.asarray(slot)-c[:2], -.08, .08)
        a[4] = .10; a[10] = -1
        s, done = step(e, s, a)
    for _ in range(8):
        c = xyz(s, name)
        a = np.zeros(11); a[:2] = np.clip(np.asarray(slot)-c[:2], -.08, .08)
        a[6] = -.10; a[10] = -1
        s, done = step(e, s, a)
    return s, done


def run(seed):
    e = make_env(); s, _ = e.reset(seed=seed, options={"object_count": 2})
    tx, ty = xyz(s, "box0")[:2]
    for q, o in zip(Q, OFF):
        s, _ = command(e, s, [tx+o[0], ty+o[1], o[2]], q, 1, 35)
        s, _ = command(e, s, [tx+o[0], ty+o[1], o[2]], q, -1, 2)
    print("seed", seed, "held",v(s,"robot","grasp_active"),v(s,"box0","grasp_active"), "initial", "b", np.round(xyz(s,"box0"),3),
          "c0",np.round(xyz(s,"cube0"),3),"c1",np.round(xyz(s,"cube1"),3))
    # First reproduce the exact validated single-cube target.
    s, done = seat(e, s, "cube0", (.60, 0.0))
    print("seat0", done, "b",np.round(xyz(s,"box0"),3),
          "c0",np.round(xyz(s,"cube0"),3),"c1",np.round(xyz(s,"cube1"),3))
    if not done:
        s, done = seat(e, s, "cube1", (.72, .22))
    print("seat1", done, "b",np.round(xyz(s,"box0"),3),
          "c0",np.round(xyz(s,"cube0"),3),"c1",np.round(xyz(s,"cube1"),3))
    # Put the still-held box on the open table area and release.
    for _ in range(20):
        b=xyz(s,"box0"); a=np.zeros(11);a[:2]=np.clip([.74-b[0],-b[1]],-.10,.10);a[10]=-1
        s, done=step(e,s,a)
    for _ in range(4):
        a=np.zeros(11);a[10]=1;s,done=step(e,s,a)
        if done: break
    print("release",done,"b",np.round(xyz(s,"box0"),3),
          "c0",np.round(xyz(s,"cube0"),3),"c1",np.round(xyz(s,"cube1"),3))
    e.close()


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]) if len(sys.argv)>1 else 0)
