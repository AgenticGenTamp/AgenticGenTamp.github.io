"""Move a receptacle aside, then place a cube at its original center."""
from env_client import make_env
import numpy as np


def pos(s, name):
    o = s.get_object_from_name(name)
    return np.array([s.get(o, f) for f in ("x", "y", "z")])


env = make_env()
s, _ = env.reset(seed=0, options={"object_count": 4})
yellow0 = pos(s, "bin_yellow").copy()
dur = (30, 31, 10, 75, 25)
cycle = sum(dur)
context = None
last_reward = None

for t in range(3 * cycle):
    stage, within = divmod(t, cycle)
    robot = s.get_object_from_name("robot")
    base = np.array([s.get(robot, f) for f in
                     ("pos_base_x", "pos_base_y", "pos_base_rot")])
    q = np.array([s.get(robot, "pos_arm_joint%d" % i) for i in range(1, 8)])
    if within == 0:
        if stage == 0:              # Align cube x with original yellow center.
            name, target = "cube4", np.array([yellow0[0], pos(s, "cube4")[1]])
        elif stage == 1:            # Clear the bin sideways from the y corridor.
            name, target = "bin_yellow", np.array([0.30, pos(s, "bin_yellow")[1]])
        else:                       # Put cube at the now-empty fixed center.
            name, target = "cube4", yellow0[:2].copy()
        source = pos(s, name)[:2]
        delta = target - source
        # Every requested leg is cardinal; retain only its dominant component.
        axis = int(abs(delta[1]) > abs(delta[0]))
        delta[1 - axis] = 0.0
        u = delta / max(1e-6, np.linalg.norm(delta))
        yaw = float(np.arctan2(u[1], u[0]))
        start_angle = float(np.arctan2(base[1], base[0]))
        end_angle = float(np.arctan2(-u[1], -u[0]))
        turn = (end_angle - start_angle + np.pi) % (2*np.pi) - np.pi
        context = (name, source, target, u, yaw, start_angle, turn)
    name, source, target, u, yaw, start_angle, turn = context
    phase = 0
    boundary = dur[0]
    while phase < 4 and within >= boundary:
        phase += 1
        boundary += dur[phase]
    if phase == 0:
        bg = np.array([np.cos(start_angle), np.sin(start_angle), yaw])
    elif phase == 1:
        frac = (within - dur[0] + 1) / dur[1]
        ang = start_angle + turn * frac
        bg = np.array([np.cos(ang), np.sin(ang), yaw])
    elif phase in (2, 3):
        bg = np.r_[source - .98*u, yaw]
    else:
        bg = np.r_[target - .965*u, yaw]
    qg = np.array([0., -.349, np.pi, -2.548, 0., -.873, np.pi/2])
    if phase >= 3:
        qg = np.array([0., 1.30, np.pi, -1.70, 0., 1.0, 0.])
    a = np.zeros(11, np.float32)
    be = bg - base
    be[2] = (be[2] + np.pi) % (2*np.pi) - np.pi
    a[:3] = np.clip(1.5*be, -.1, .1)
    a[3:10] = np.clip(1.5*(qg-q), -.1, .1)
    s, reward, term, trunc, _ = env.step(a)
    dfix = np.linalg.norm(pos(s, "cube4") - yellow0)
    dnow = np.linalg.norm(pos(s, "cube4") - pos(s, "bin_yellow"))
    if reward != last_reward or within in (0, cycle-1) or dfix < .06:
        print(t+1, stage, within, reward, "fixed", round(float(dfix), 4),
              "current", round(float(dnow), 4), "cube", np.round(pos(s,"cube4")[:2],3),
              "bin", np.round(pos(s,"bin_yellow")[:2],3), "term", term)
    last_reward = reward
    if term or trunc:
        break
env.close()
