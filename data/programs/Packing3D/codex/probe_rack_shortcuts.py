"""Black-box probes for manipulating the rack and pushing parts."""
import numpy as np

from env_client import make_env


PICK = np.array([0., .265462, -np.pi, -2.031427, 0., -.758715, np.pi/2])
OFFSET = np.array([.4803523, .008578])


def g(s, name, feature):
    return s.get(s.get_object_from_name(name), feature)


def move(env, s, base, joints, grip=0., steps=30):
    for _ in range(steps):
        a = np.zeros(11, np.float32)
        a[0] = np.clip(base[0] - g(s, "robot", "pos_base_x"), -.2, .2)
        a[1] = np.clip(base[1] - g(s, "robot", "pos_base_y"), -.2, .2)
        a[2] = np.clip(-g(s, "robot", "pos_base_rot"), -.2, .2)
        for i in range(7):
            a[3+i] = np.clip(joints[i] - g(s, "robot", f"joint_{i+1}"), -.2, .2)
        a[10] = grip
        s, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
        if max(abs(g(s, "robot", "pos_base_x")-base[0]),
               abs(g(s, "robot", "pos_base_y")-base[1]),
               *(abs(g(s, "robot", f"joint_{i+1}")-joints[i]) for i in range(7))) < .004:
            break
    return s


def rack_grasp(offset):
    env = make_env()
    try:
        s, _ = env.reset(seed=0, options={"object_count": 3})
        r0 = np.array([g(s, "rack", "pose_x"), g(s, "rack", "pose_y"), g(s, "rack", "pose_z")])
        base = r0[:2] - OFFSET + np.asarray(offset)
        s = move(env, s, base, PICK, 1.)
        for _ in range(3):
            a = np.zeros(11, np.float32); a[10] = -1.
            s, _, _, _, _ = env.step(a)
        ga = (g(s, "robot", "grasp_active"), g(s, "rack", "grasp_active"))
        lifted = PICK.copy(); lifted[1] -= .15
        s = move(env, s, base, lifted, 0.)
        r1 = np.array([g(s, "rack", "pose_x"), g(s, "rack", "pose_y"), g(s, "rack", "pose_z")])
        return ga, r1-r0
    finally:
        env.close()


def push_object(name, approach_offset, push_delta, seed=0, closed=False):
    """Put the open fingers at an object's height then translate the mobile base."""
    env = make_env()
    try:
        s, _ = env.reset(seed=seed, options={"object_count": 3})
        before = {n: np.array([g(s, n, "pose_x"), g(s, n, "pose_y"), g(s, n, "pose_z")])
                  for n in s.get_object_names() if n != "robot"}
        target = before[name][:2]
        base = target - OFFSET + np.asarray(approach_offset)
        s = move(env, s, base, PICK, 1.)
        if closed:
            for _ in range(3):
                a = np.zeros(11, np.float32); a[10] = -1.
                s, _, _, _, _ = env.step(a)
        grasp = g(s, "robot", "grasp_active")
        destination = base + np.asarray(push_delta)
        s = move(env, s, destination, PICK, 0. if closed else 1., steps=15)
        after = {n: np.array([g(s, n, "pose_x"), g(s, n, "pose_y"), g(s, n, "pose_z")])
                 for n in before}
        return grasp, {n: np.round(after[n]-before[n], 5) for n in before
                       if np.linalg.norm(after[n]-before[n]) > 1e-5}
    finally:
        env.close()


def held_object_push(source, target, seed=0):
    """Grasp one part and sweep it horizontally through another object."""
    env = make_env()
    try:
        s, _ = env.reset(seed=seed, options={"object_count": 3})
        names = [n for n in s.get_object_names() if n != "robot"]
        before = {n: np.array([g(s, n, "pose_x"), g(s, n, "pose_y"), g(s, n, "pose_z")])
                  for n in names}
        base = before[source][:2] - OFFSET
        s = move(env, s, base, PICK, 1.)
        for _ in range(3):
            a = np.zeros(11, np.float32); a[10] = -1.
            s, _, _, _, _ = env.step(a)
        grasp = g(s, "robot", "grasp_active")
        destination = before[target][:2] - OFFSET
        s = move(env, s, destination, PICK, 0., steps=15)
        after = {n: np.array([g(s, n, "pose_x"), g(s, n, "pose_y"), g(s, n, "pose_z")])
                 for n in names}
        return grasp, {n: np.round(after[n]-before[n], 5) for n in names
                       if np.linalg.norm(after[n]-before[n]) > 1e-5}
    finally:
        env.close()


def chassis_ram(target, seed=0):
    """Drive the mobile base through an object's initial footprint."""
    env = make_env()
    try:
        s, _ = env.reset(seed=seed, options={"object_count": 3})
        names = [n for n in s.get_object_names() if n != "robot"]
        before = {n: np.array([g(s, n, "pose_x"), g(s, n, "pose_y"), g(s, n, "pose_z")])
                  for n in names}
        joints = np.array([g(s, "robot", f"joint_{i+1}") for i in range(7)])
        s = move(env, s, np.array([-.12, before[target][1]]), joints, 1.)
        s = move(env, s, np.array([.45, before[target][1]]), joints, 1., steps=15)
        after = {n: np.array([g(s, n, "pose_x"), g(s, n, "pose_y"), g(s, n, "pose_z")])
                 for n in names}
        base = np.array([g(s, "robot", "pos_base_x"), g(s, "robot", "pos_base_y")])
        return np.round(base, 5), {n: np.round(after[n]-before[n], 5) for n in names
                                  if np.linalg.norm(after[n]-before[n]) > 1e-5}
    finally:
        env.close()


if __name__ == "__main__":
    for dx in (-.08, -.04, 0., .04, .08):
        for dy in (-.12, -.06, 0., .06, .12):
            ga, d = rack_grasp((dx, dy))
            print((dx, dy), "grasp", ga, "rack_delta", np.round(d, 5))
