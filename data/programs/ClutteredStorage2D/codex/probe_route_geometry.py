"""Log black-box shelf interaction geometry for the current policy."""

import sys

from approach import GeneratedApproach
from env_client import make_env


def f(state, obj, feature):
    return float(state.get(obj, feature))


def run(seed):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 1})
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    robot = state.get_object_from_name("robot")
    block = state.get_object_from_name("block0")
    shelf = state.get_objects(env.observation_space.get_type("shelf"))[0]
    print("SHELF", *(round(f(state, shelf, q), 5) for q in
          ("x", "y", "width", "height", "x1", "y1", "width1", "height1")))
    previous = None
    for step in range(env.max_steps):
        slot_x = f(state, shelf, "x1") + f(state, shelf, "width1") / 2
        policy.place = (slot_x, f(state, shelf, "y1") + .18)
        policy.preplace = (slot_x, 2.29)
        stage = policy.stage
        # Experimental variant: preserve a vertical block through both orient
        # stages.  The stock policy makes it horizontal, where its 0.28 width
        # leaves almost no jamb clearance.
        if policy.stage == "preplace":
            # Align with the narrow opening while still low, then rise.  A
            # diagonal shortcut lets the long vertical block strike the solid
            # underside before it reaches the opening.
            bx, by = f(state, block, "x"), f(state, block, "y")
            ex, ey = policy.preplace[0]-bx, policy.preplace[1]-by
            if abs(ex) >= .008:
                action = policy._action(dx=ex, vac=1.0)
            elif abs(ey) >= .008:
                action = policy._action(dy=ey, vac=1.0)
            else:
                policy.stage = "insert"
                policy.last_insert_y = by
                policy.insert_stall = 0
                policy.insert_toggle = 0
                action = policy._action(vac=1.0)
        elif policy.stage == "renavigate" and policy.regrasped:
            # Move around the released vertical block rather than diagonally
            # through it: first below it, then horizontally under its center.
            bx, by = f(state, block, "x"), f(state, block, "y")
            rx, ry = f(state, robot, "x"), f(state, robot, "y")
            rt, arm = f(state, robot, "theta"), f(state, robot, "arm_joint")
            gy = by - 0.54
            if arm > 0.21:
                action = policy._action(da=0.2-arm)
            elif ry > gy + 0.008:
                action = policy._action(dy=gy-ry)
            elif abs(rx-bx) > 0.008:
                action = policy._action(dx=bx-rx)
            else:
                et = (3.141592653589793/2-rt+3.141592653589793) % (2*3.141592653589793)-3.141592653589793
                if abs(et) > .012:
                    action = policy._action(dt=et)
                else:
                    policy.stage = "reextend"
                    policy.last_arm = arm
                    policy.stall_ticks = 0
                    action = policy._action()
        elif policy.stage == "retract" and not policy.regrasped:
            # Keep enough radial clearance that a 0.28-long vertical block does
            # not overlap the circular base when released for regrasping.
            arm = f(state, robot, "arm_joint")
            if arm > 0.50:
                action = policy._action(da=0.50 - arm, vac=1.0)
            else:
                policy.stage = "orient"
                action = policy._action(vac=1.0)
        elif policy.stage == "orient":
            bt = f(state, block, "theta")
            rt = f(state, robot, "theta")
            offset = (bt - rt + 3.141592653589793) % (2 * 3.141592653589793) - 3.141592653589793
            desired = 3.141592653589793 / 2 - offset
            error = (desired - rt + 3.141592653589793) % (2 * 3.141592653589793) - 3.141592653589793
            if abs(error) >= 0.025:
                action = policy._action(dt=error, vac=1.0)
            elif not policy.regrasped:
                policy.stage = "drop_for_regrasp"
                action = policy._action(vac=0.0)
            else:
                policy.stage = "preplace"
                action = policy._action(vac=1.0)
        else:
            action = policy.get_action(state)
        if stage != previous or step % 25 == 0:
            print(step, stage, "R", *(round(f(state, robot, q), 4) for q in
                  ("x", "y", "theta", "arm_joint", "vacuum")),
                  "B", *(round(f(state, block, q), 4) for q in ("x", "y", "theta")),
                  "A", *(round(float(x), 4) for x in action))
            previous = stage
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("END", step + 1, terminated, truncated, reward, info,
                  "B", *(round(f(state, block, q), 5) for q in ("x", "y", "theta")))
            break
    env.close()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
