import warnings
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv

TABLE_TOP = 0.1
RKEYS = ["x", "y", "theta", "arm_joint", "finger_gap", "base_radius", "arm_length",
         "finger_height", "finger_width", "gripper_base_width", "gripper_base_height"]


def _find(state, type_name):
    return [state.get_object_from_name(n) for n in state.get_object_names()
            if state.get_object_from_name(n).type.name == type_name]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Ctl:
    def __init__(self, action_space):
        self.action_space = action_space

    def _parse(self, state):
        robot = _find(state, "kin_robot")[0]
        r = {k: float(state.get(robot, k)) for k in RKEYS}
        return r, blocks(state)

    def _action(self, r, tx=None, ty=None, tth=None, tarm=None, tgap=None):
        a = np.zeros(5)
        if tx is not None: a[0] = np.clip(tx - r["x"], -0.05, 0.05)
        if ty is not None: a[1] = np.clip(ty - r["y"], -0.05, 0.05)
        if tth is not None: a[2] = np.clip(_wrap(tth - r["theta"]), -np.pi / 16, np.pi / 16)
        if tarm is not None: a[3] = np.clip(tarm - r["arm_joint"], -0.1, 0.1)
        if tgap is not None: a[4] = np.clip(tgap - r["finger_gap"], -0.02, 0.02)
        lo, hi = self.action_space.low * 0.999, self.action_space.high * 0.999
        return np.clip(a, lo, hi).astype(np.float64)
warnings.filterwarnings("ignore")
WALL_R = 1 + np.sqrt(5)


def make(n, seed):
    env = VariableObjectCountEnv(
        constant_object_env_path="kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
        count_kwarg="num_obstructions", count_object_prefix="obstruction",
        design_counts=[n], eval_counts=[n])
    state, info = env.reset(seed=seed, options={"object_count": n})
    ap = Ctl(env.action_space)
    return env, state, ap


def set_block(env, state, name, **kw):
    st = state.copy()
    obj = [o for o in st if o.name == name][0]
    for k, v in kw.items():
        st.set(obj, k, v)
    s2, _ = env.reset(options={"init_state": st})
    return s2


def blocks(state):
    out = {}
    for t in ("target_block", "dyn_rectangle"):
        for o in _find(state, t):
            out[o.name] = {k: float(state.get(o, k)) for k in ["x", "y", "theta", "width", "height"]}
    return out


def xext(o):
    c, s_ = np.cos(o["theta"]), np.sin(o["theta"])
    hw, hh = o["width"] / 2, o["height"] / 2
    xs = [o["x"] + sx * hw * c - sy * hh * s_ for sx in (-1, 1) for sy in (-1, 1)]
    return min(xs), max(xs)


def wall_gaps(o):
    l, r = xext(o)
    return l, WALL_R - r


class Sim:
    def __init__(self, env, state, ap):
        self.env, self.state, self.ap = env, state, ap
        self.t = 0

    def P(self):
        return self.ap._parse(self.state)

    def step(self, a):
        self.state, *_ = self.env.step(a)
        self.t += 1

    def go(self, tx=None, ty=None, tth=None, tarm=None, tgap=None, n=200, tol=2e-3, cb=None):
        for _ in range(n):
            r = self.P()[0]
            done = True
            if tx is not None and abs(tx - r["x"]) > tol: done = False
            if ty is not None and abs(ty - r["y"]) > tol: done = False
            if tth is not None and abs(tth - r["theta"]) > 1e-3: done = False
            if tarm is not None and abs(tarm - r["arm_joint"]) > tol: done = False
            if tgap is not None and abs(tgap - r["finger_gap"]) > tol: done = False
            if done:
                return True
            self.step(self.ap._action(r, tx=tx, ty=ty, tth=tth, tarm=tarm, tgap=tgap))
            if cb: cb(self)
        return False

    def robot_str(self):
        r = self.P()[0]
        return f"R({r['x']:.3f},{r['y']:.3f},th={r['theta']:.2f},L={r['arm_joint']:.3f},g={r['finger_gap']:.3f})"


def inner(env, n):
    return env._backend_for(n)._object_centric_env


def shapes_bb(env, n):
    rb = inner(env, n).robot
    out = {}
    for nm in ['_base_shape', '_gripper_base_shape', '_arm_shape', '_left_finger_shape', '_right_finger_shape']:
        bb = getattr(rb, nm).cache_bb()
        out[nm] = tuple(round(v, 3) for v in (bb.left, bb.bottom, bb.right, bb.top))
    return out
