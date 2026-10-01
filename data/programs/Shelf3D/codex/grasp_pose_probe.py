"""Move the one-cube robot to an arm posture and render/check contact."""

import sys
import numpy as np

from env_client import make_env


def val(state, name, feature):
    return state.get(state.get_object_from_name(name), feature)


def main():
    target = np.array([float(x) for x in sys.argv[1:8]])
    label = sys.argv[8] if len(sys.argv) > 8 else "pose_probe"
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube = next(n for n in state.get_object_names() if n.startswith("cube"))
    start = np.array([val(state, cube, f) for f in ("x", "y", "z")])
    for _ in range(140):
        q = np.array([val(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])
        error = target - q
        if np.max(np.abs(error)) < 0.02:
            break
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = np.clip(0.15 * error, -0.1, 0.1)
        state, *_ = env.step(action)
    for _ in range(5):
        state, *_ = env.step(np.zeros(11, dtype=np.float32))
    end = np.array([val(state, cube, f) for f in ("x", "y", "z")])
    q = np.array([val(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])
    print("target", np.round(target, 3), "actual", np.round(q, 3),
          "cube", np.round(start, 3), "->", np.round(end, 3),
          "render", env.render_state(state=state, label=label))
    env.close()


if __name__ == "__main__":
    main()
