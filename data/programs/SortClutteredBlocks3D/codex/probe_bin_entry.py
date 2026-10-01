"""Center one cube in its matching bin, retract, and observe settling."""
import numpy as np

from env_client import make_env
from edge_rake_probe import Q, action_to, cubes, get, robot, settle_to


def run(seed=0):
    env = make_env()
    state, _ = env.reset(seed=seed, options={"object_count": 4})
    initial = cubes(state)
    name = min(initial, key=lambda n: initial[n][1])
    bin_name = "bin_red"
    goal = get(state, bin_name, ["x", "y", "z"])

    # Reach the pile from the unobstructed left side with the arm folded.
    state = settle_to(env, state, [.99, .80, np.pi], None, 45, 0)
    state = settle_to(env, state, [-1.0, .80, np.pi], None, 55, 0)
    state = settle_to(env, state, [-1.0, .80, 0.0], None, 45, 0)
    ybase = float(initial[name][1] + .042)
    state = settle_to(env, state, [-1.0, ybase, 0.0], None, 45, 0)
    state = settle_to(env, state, [-1.0, ybase, 0.0], Q, 130, 0)

    # Approach slowly until any cube moves.
    hit = None
    for _ in range(50):
        action = action_to(state, [-.80, ybase, 0.0], Q, 0)
        action[0] = min(action[0], .012)
        state, reward, term, trunc, _ = env.step(action)
        if max(np.linalg.norm(v - initial[n]) for n, v in cubes(state).items()) > .001:
            hit = robot(state)[:3].copy()
            break
    if hit is None:
        print("NO_CONTACT")
        env.close()
        return

    # The positive joint-1 arc carries the exposed red cube toward -y and -x.
    # Move the base in y using observed error so the second half of the arc
    # centers rather than overshooting the receptacle.
    qtarget = Q.copy()
    qtarget[0] = .82
    base_target = hit.copy()
    best = (99.0, None)
    for step in range(150):
        cube = cubes(state)[name]
        # Shifting the robot +y shifts the contact +y.  Limit changes so the
        # fingertip does not lose the cube abruptly.
        base_target[1] += float(np.clip(1.4 * (goal[1] - cube[1]), -.012, .012))
        state, reward, term, trunc, _ = env.step(action_to(state, base_target, qtarget, 0))
        cube = cubes(state)[name]
        distance = float(np.linalg.norm(cube[:2] - goal[:2]))
        if distance < best[0]:
            best = (distance, cube.copy())
        if step % 10 == 0 or distance < .025 or term:
            print("PUSH", step, "reward", reward, "term", term,
                  "dxy", round(distance, 4), "cube", np.round(cube, 4),
                  "bin", np.round(get(state, bin_name, ["x", "y", "z"]), 4),
                  "base", np.round(robot(state)[:3], 4), "q1", round(robot(state)[3], 4))
        if distance < .018 or term or trunc:
            break

    # Withdraw left while holding the arm fixed, then fold it.  Report whether
    # the cube remains centered and whether any reward/termination arrives.
    retract = robot(state)[:3].copy()
    retract[0] = -1.10
    for step in range(160):
        qgoal = Q if step < 70 else np.array([0., -.349, np.pi, -2.548, 0., -.873, np.pi/2])
        state, reward, term, trunc, _ = env.step(action_to(state, retract, qgoal, 0))
        if step % 10 == 0 or reward != -1.0 or term:
            cube = cubes(state)[name]
            print("SETTLE", step, "reward", reward, "term", term,
                  "dxy", round(float(np.linalg.norm(cube[:2] - get(state, bin_name, ["x", "y"])[:2])), 4),
                  "cube", np.round(cube, 4), "bin", np.round(get(state, bin_name, ["x", "y", "z"]), 4))
        if term or trunc:
            break
    print("BEST", round(best[0], 5), None if best[1] is None else np.round(best[1], 5))
    env.close()


if __name__ == "__main__":
    run()
