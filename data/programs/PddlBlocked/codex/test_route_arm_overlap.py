"""A/B test unfolding the arm during local approach translation."""
import sys

from env_client import make_env
from approach import GeneratedApproach


class ArmOverlapApproach(GeneratedApproach):
    def motion(self, state, target, grip=0., lift=False, arm=True, q4add=0.):
        # Only alter stage 0.  The stock policy suppresses arm motion on its
        # collision-avoidance approach waypoints; test doing that setup in
        # parallel with the base drive.
        if self.stage == 0:
            arm = True
        return super().motion(state, target, grip, lift, arm, q4add)


def run(policy_type, seed):
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = policy_type(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    initial_stage = policy.stage
    for step in range(env.max_steps):
        state, _, terminated, truncated, _ = env.step(policy.get_action(state))
        if terminated or truncated:
            break
    env.close()
    return terminated, step + 1, initial_stage


start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stop = int(sys.argv[2]) if len(sys.argv) > 2 else start + 200
changed = []
regressions = []
for seed in range(start, stop):
    base = run(GeneratedApproach, seed)
    candidate = run(ArmOverlapApproach, seed)
    if base != candidate:
        changed.append((seed, base, candidate))
    if base[0] and (not candidate[0] or candidate[1] > base[1]):
        regressions.append((seed, base, candidate))
print("changed", changed[:30])
print("changed_count", len(changed), "regressions", regressions)
