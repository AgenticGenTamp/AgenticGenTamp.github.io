"""Low-level helpers for driving the TidyBot env (servo loops)."""
import numpy as np
import kin

J = ['pos_arm_joint%d' % i for i in range(1, 8)]
ARM_GAIN = 1.5     # a is delta on position target (q_meas + a); speed saturates ~0.025 rad/step
BASE_GAIN = 1.0 / 0.87


class Bot:
    def __init__(self, env, obs):
        self.env = env
        self.obs = obs
        self.R = obs.get_object_from_name('robot')
        self.grip = 0.0
        self.nsteps = 0

    # ---- state
    def q(self):
        return np.array([self.obs.get(self.R, j) for j in J])

    def base(self):
        o, R = self.obs, self.R
        return np.array([o.get(R, 'pos_base_x'), o.get(R, 'pos_base_y'), o.get(R, 'pos_base_rot')])

    def gripper(self):
        return self.obs.get(self.R, 'pos_gripper')

    def obj(self, name, feats=('x', 'y', 'z')):
        o = self.obs.get_object_from_name(name)
        return np.array([self.obs.get(o, f) for f in feats])

    def tip(self):
        b = self.base()
        return kin.fk(b[0], b[1], b[2], self.q())

    # ---- actions
    def step(self, a):
        a = np.asarray(a, np.float32)
        self.obs, r, term, trunc, info = self.env.step(a)
        self.nsteps += 1
        return r

    def act(self, q_t=None, base_t=None, grip=None):
        a = np.zeros(11, np.float32)
        if grip is not None:
            self.grip = grip
        a[10] = self.grip
        if q_t is not None:
            dq = kin.wrap_to(q_t, self.q()) - self.q()
            a[3:10] = np.clip(ARM_GAIN * dq, -0.1, 0.1)
        if base_t is not None:
            b = self.base()
            d = np.array(base_t, float) - b
            d[2] = (d[2] + np.pi) % (2 * np.pi) - np.pi
            a[0:2] = np.clip(BASE_GAIN * d[0:2], -0.1, 0.1)
            a[2] = np.clip(d[2], -0.1, 0.1)
        return self.step(a)

    def goto(self, q_t=None, base_t=None, grip=None, tol=0.005, max_steps=300, min_steps=0):
        for k in range(max_steps):
            self.act(q_t, base_t, grip)
            done = k >= min_steps
            if q_t is not None:
                done &= np.max(np.abs(kin.wrap_to(q_t, self.q()) - self.q())) < tol
            if base_t is not None:
                b = self.base()
                d = np.array(base_t) - b
                d[2] = (d[2] + np.pi) % (2 * np.pi) - np.pi
                done &= np.max(np.abs(d)) < tol
            if done:
                return True
        return False

    def wait(self, n, grip=None):
        for _ in range(n):
            self.act(q_t=self.q(), grip=grip)


def cube_yaw(bot, name):
    c = bot.obj(name, ('qw', 'qx', 'qy', 'qz'))
    return 2 * np.arctan2(c[3], c[0])


def move_tip(bot, p, yaw=None, yaw_sym=np.pi / 2, tol=0.002, max_steps=300):
    """IK (gripper down) for current base pose, then servo the arm there."""
    q, ok, e = kin.ik_multi(bot.base(), bot.q(), p, 'down', yaw=yaw, yaw_sym=yaw_sym)
    bot.goto(q_t=q, tol=tol, max_steps=max_steps)
    return ok


def pick(bot, name, standoff=0.5, approach_yaw=None, lift=0.15):
    """Drive base next to the cube, grasp from above, lift. Returns True if held."""
    c = bot.obj(name)
    b = bot.base()
    if approach_yaw is None:  # approach from robot's current side
        approach_yaw = np.arctan2(b[1] - c[1], b[0] - c[0])
    base_t = [c[0] + standoff * np.cos(approach_yaw), c[1] + standoff * np.sin(approach_yaw),
              approach_yaw + np.pi]
    bot.goto(base_t=base_t, grip=0.0, tol=0.005, max_steps=150)
    y = cube_yaw(bot, name)
    move_tip(bot, c + [0, 0, 0.10], yaw=y)
    move_tip(bot, c + [0, 0, 0.0], yaw=y)
    bot.wait(12, grip=1.0)
    move_tip(bot, c + [0, 0, lift], yaw=y)
    return bot.obj(name)[2] > c[2] + 0.5 * lift


def place(bot, name, xy, standoff=0.5, drop_z=0.012):
    """Carry held cube to world xy (base drives, arm keeps pose), lower, release."""
    b = bot.base()
    ang = np.arctan2(b[1] - xy[1], b[0] - xy[0])
    base_t = [xy[0] + standoff * np.cos(ang), xy[1] + standoff * np.sin(ang), ang + np.pi]
    bot.goto(base_t=base_t, tol=0.005, max_steps=150)
    move_tip(bot, [xy[0], xy[1], 0.10])
    move_tip(bot, [xy[0], xy[1], drop_z])
    bot.wait(8, grip=0.0)
    move_tip(bot, [xy[0], xy[1], 0.15])
    return bot.obj(name)
