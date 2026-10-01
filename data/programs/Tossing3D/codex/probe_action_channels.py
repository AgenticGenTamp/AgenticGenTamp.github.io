"""Empirically identify Tossing3D action channels from joint responses."""
import numpy as np
from env_client import make_env


def robot_vec(state):
    r = state.get_object_from_name("robot")
    q = np.array([state.get(r, f"pos_arm_joint{i}") for i in range(1, 8)], float)
    v = np.array([state.get(r, f"vel_arm_joint{i}") for i in range(1, 8)], float)
    b = np.array([state.get(r, "pos_base_x"), state.get(r, "pos_base_y"),
                  state.get(r, "pos_base_rot")], float)
    return b, q, v, float(state.get(r, "pos_gripper"))


def trial(env, index, value, steps=1):
    s, _ = env.reset(seed=3)
    initial = robot_vec(s)
    rewards = []
    for _ in range(steps):
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        a[index] = value
        s, rew, term, trunc, _ = env.step(a)
        rewards.append(rew)
    final = robot_vec(s)
    return initial, final, rewards


np.set_printoptions(precision=5, suppress=True, linewidth=160)
env = make_env()
for steps in (1, 5):
    print("STEPS", steps)
    for idx in range(3, 18):
        value = 0.08 if idx < 10 else (1.0 if idx == 10 else 1.0)
        ini, fin, rewards = trial(env, idx, value, steps)
        print(f"i={idx:2d} val={value:g} db={fin[0]-ini[0]} dq={fin[1]-ini[1]} v={fin[2]} dg={fin[3]-ini[3]:.5f}", flush=True)
env.close()

# Same physical joint commanded through both seven-channel banks.  This tests
# whether the trailing bank is independent and whether effects superpose.
env = make_env()
print("COMBINED joint1 one-step", flush=True)
for pos, vel in ((0.08, 1.0), (0.08, -1.0), (-0.08, 1.0),
                 (0.0, 0.1), (0.0, 3.0), (0.0, 6.0)):
    s, _ = env.reset(seed=3)
    ini = robot_vec(s)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[3], a[11] = pos, vel
    s, _, _, _, _ = env.step(a)
    fin = robot_vec(s)
    print(f"pos={pos:+.2f} vel={vel:+.1f} dq1={fin[1][0]-ini[1][0]:+.5f} observed_v1={fin[2][0]:+.5f}", flush=True)
env.close()
