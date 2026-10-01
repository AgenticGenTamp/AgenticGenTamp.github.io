"""Test tighter box-tool staging without modifying the submitted policy."""
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


class Candidate(GeneratedApproach):
    gap = .10
    stage_blocks = 1

    def get_action(self, state):
        old_phase = self.phase
        action = super().get_action(state)
        if old_phase == "tool_stage" and self.phase == "tool_sweep" and self.stage_blocks > 1:
            self.stage_blocks -= 1
            self.phase = "tool_stage"
            self.phase_ticks = 0
        # Replace tool-stage translation with a closer point behind the cube.
        if old_phase == "tool_stage" and self.phase == "tool_stage":
            c = np.array([self.g(state, self.tool_cube, "pose_x"),
                          self.g(state, self.tool_cube, "pose_y")])
            u = np.array([.6-c[0], -c[1]])
            u /= max(np.linalg.norm(u), 1e-6)
            side = c-u*self.gap
            p = np.array([self.g(state, "box0", "pose_x"),
                          self.g(state, "box0", "pose_y")])
            bx = self.g(state, "robot", "pos_base_x")
            by = self.g(state, "robot", "pos_base_y")
            action[:2] = np.clip([bx+side[0]-p[0], by+side[1]-p[1]], -.2, .2)
        return action


for gap in (.18,):
    Candidate.gap = gap
    for blocks in (2, 3):
      for seed in (0, 2, 3, 7):
        Candidate.stage_blocks = blocks
        env = make_env(); state, info = env.reset(seed=seed, options={"object_count": 1})
        policy = Candidate(env.action_space, env.observation_space, {})
        policy.reset(state, info)
        term = False
        for step in range(300):
            state, _, term, trunc, _ = env.step(policy.get_action(state))
            if term or trunc: break
        cube = [policy.g(state, "cube0", "pose_"+c) for c in "xyz"]
        print("gap", gap, "blocks", blocks, "seed", seed, "term", term, "step", step,
              "phase", policy.phase, "cube", np.round(cube, 2))
        env.close()
