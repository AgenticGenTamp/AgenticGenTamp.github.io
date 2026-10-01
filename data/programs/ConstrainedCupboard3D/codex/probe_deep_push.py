"""Probe whether continuing the known floor-pusher into the cupboard scores."""
import numpy as np

from env_client import make_env
from approach import GeneratedApproach


class DeepPush(GeneratedApproach):
    """The submitted policy with its conservative cupboard-mouth stop removed."""

    def get_action(self, state):
        if self.step < 120:
            return super().get_action(state)
        action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        cycle = 62
        phase = (self.step - 120) % cycle
        self.step += 1
        rod = self.rods[0]
        robot = state.get_object_from_name("robot")
        rx = self._v(state, rod, "x")
        ry = self._v(state, rod, "y")
        side = .10 if ry < self.slot_y[0] else -.10
        if phase == 0:
            self.stroke_x = rx
        if phase < 25:
            action[-1] = 1.
            self._base(state, robot, action, rx - .80, ry - 5 * side, 0.)
        elif phase < 37:
            action[-1] = 1.
            self._base(state, robot, action, rx - .80, ry + side, 0.)
        elif phase < 47:
            self._base(state, robot, action, rx - .80, ry + side, 0.)
        else:
            self._base(state, robot, action, min(2.15, self.stroke_x + .80),
                       ry + side, 0.)
        self._joints(state, robot, action, self.pick_q)
        return action


def run(seed):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 1})
    policy = DeepPush(env.action_space, env.observation_space,
                      env.make_primitives())
    policy.reset(state, info)
    rod = policy.rods[0]
    initial = tuple(float(state.get(rod, f)) for f in ("x", "y", "z"))
    best_reward = -1e9
    events = []
    for step in range(env.max_steps):
        state, reward, term, trunc, info = env.step(policy.get_action(state))
        if reward > best_reward:
            best_reward = reward
            pose = tuple(float(state.get(rod, f)) for f in ("x", "y", "z"))
            events.append((step + 1, reward, pose, term, trunc))
        if term or trunc:
            break
    final = tuple(float(state.get(rod, f)) for f in ("x", "y", "z"))
    print({"seed": seed, "target_y": policy.slot_y[0], "initial": initial,
           "final": final, "steps": step + 1, "best_reward": best_reward,
           "terminated": term, "truncated": trunc, "events": events})
    env.close()


if __name__ == "__main__":
    run(3)
