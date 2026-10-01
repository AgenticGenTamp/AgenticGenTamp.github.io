"""Black-box probes for ClutteredStorage2D shelf/release geometry."""

import math
import sys

import numpy as np

from env_client import make_env


def val(state, obj, key):
    return float(state.get(obj, key))


def summarize(seed, count=None):
    env = make_env()
    options = {} if count is None else {"object_count": count}
    state, info = env.reset(seed=seed, options=options)
    robot = state.get_objects(env.observation_space.get_type("crv_robot"))[0]
    shelf = state.get_objects(env.observation_space.get_type("shelf"))[0]
    blocks = state.get_objects(env.observation_space.get_type("target_block"))
    print("seed", seed, "count_request", count, "info", info)
    print(" robot", {k: round(val(state, robot, k), 4) for k in
                     ("x", "y", "theta", "base_radius", "arm_joint", "arm_length",
                      "vacuum", "gripper_height", "gripper_width")})
    print(" shelf", {k: round(val(state, shelf, k), 4) for k in
                     ("x", "y", "theta", "width", "height", "x1", "y1", "theta1",
                      "width1", "height1")})
    for block in blocks:
        print(" block", block.name, {k: round(val(state, block, k), 4)
                                     for k in ("x", "y", "theta", "width", "height")})
    env.close()


def move_to_grasp_and_target(seed, target_mode):
    """Simple feedback controller, only for experiments with one block."""
    env = make_env()
    state, _ = env.reset(seed=seed, options={"object_count": 1})
    typ = env.observation_space
    holding = False
    grasp_offset = None
    released = False
    for step in range(700):
        robot = state.get_objects(typ.get_type("crv_robot"))[0]
        shelf = state.get_objects(typ.get_type("shelf"))[0]
        block = state.get_objects(typ.get_type("target_block"))[0]
        rx, ry = val(state, robot, "x"), val(state, robot, "y")
        theta = val(state, robot, "theta")
        arm = val(state, robot, "arm_joint")
        bx, by = val(state, block, "x"), val(state, block, "y")
        sx, sy = val(state, shelf, "x"), val(state, shelf, "y")
        # Gripper center: base + base radius + extension + half its long dimension.
        reach = val(state, robot, "base_radius") + arm + val(state, robot, "gripper_width") / 2
        gx, gy = rx + reach * math.cos(theta), ry + reach * math.sin(theta)

        if not holding:
            target = (bx, by)
        else:
            if target_mode == "shelf0":
                target = (sx, sy)
            elif target_mode == "shelf1":
                target = (val(state, shelf, "x1"), val(state, shelf, "y1"))
            elif target_mode == "mid":
                target = ((sx + val(state, shelf, "x1")) / 2,
                          (sy + val(state, shelf, "y1")) / 2)
            elif target_mode == "offset":
                target = (sx + 0.1 * math.cos(val(state, shelf, "theta")),
                          sy + 0.1 * math.sin(val(state, shelf, "theta")))
            elif target_mode == "right":
                # Shelf coordinates appear to be lower-left, not center.
                target = (sx + val(state, shelf, "width") - 0.25, sy + 0.08)
            elif target_mode == "left":
                target = (sx + 0.25, sy + 0.08)
            elif target_mode == "safe":
                target = (sx + val(state, shelf, "width") - 0.7, sy + 0.08)
            elif target_mode == "goal":
                # State x coordinates use a [0, width] world frame; x1 is slot inset.
                target = (val(state, shelf, "x1") + val(state, shelf, "width1")/2,
                          val(state, shelf, "y1"))
            else:
                raise ValueError(target_mode)

        ex, ey = target[0] - gx, target[1] - gy
        dist = math.hypot(ex, ey)
        contact_ready = False
        vac = 1.0 if holding and not released else 0.0
        if not holding:
            vx, vy = bx - rx, by - ry
            angle = math.atan2(vy, vx)
            da = (angle - theta + math.pi) % (2 * math.pi) - math.pi
            radial_support = (abs(math.cos(angle-val(state, block, "theta")))*val(state, block, "width")/2 +
                              abs(math.sin(angle-val(state, block, "theta")))*val(state, block, "height")/2)
            desired_arm = (math.hypot(vx, vy) - val(state, robot, "base_radius") -
                           val(state, robot, "gripper_width")/2 - radial_support)
            contact_ready = abs(desired_arm-arm) < 0.003 and abs(da) < 0.015
            vac = 1.0 if contact_ready else 0.0
            # Face the block, then put the gripper center on it by translating or extending.
            if abs(da) > 0.015:
                action = np.array([0, 0, np.clip(da, -0.196, 0.196), -0.1, vac], dtype=np.float32)
            elif desired_arm > 0.75:
                base_move = min(0.05, desired_arm - 0.65)
                action = np.array([base_move * math.cos(angle), base_move * math.sin(angle), 0, 0, vac], dtype=np.float32)
            else:
                derr = desired_arm - arm
                if abs(derr) < 0.003:
                    # Diagnostic jiggle: an attached block should follow this.
                    action = np.array([0.01, 0, 0, 0, vac], dtype=np.float32)
                else:
                    action = np.array([0, 0, 0, np.clip(derr, -0.1, 0.1), vac], dtype=np.float32)
        else:
            # Keep the block horizontal while inserting; vacuum preserves its angle offset.
            desired_theta = -grasp_offset
            da = (desired_theta - theta + math.pi) % (2 * math.pi) - math.pi
            insert_arm = 0.75
            insert_reach = val(state, robot, "base_radius") + insert_arm + val(state, robot, "gripper_width")/2
            desired_rx = target[0] - insert_reach*math.cos(desired_theta)
            desired_ry = target[1] - insert_reach*math.sin(desired_theta)
            pose_error = math.hypot(desired_rx-rx, desired_ry-ry)
            if arm > 0.21:
                action = np.array([0, 0, 0, -0.1, vac], dtype=np.float32)
            elif abs(da) > 0.015:
                action = np.array([0, 0, np.clip(da, -0.196, 0.196), 0, vac], dtype=np.float32)
            elif pose_error > 0.015:
                action = np.array([np.clip(desired_rx-rx, -0.05, 0.05),
                                   np.clip(desired_ry-ry, -0.05, 0.05), 0, 0, vac], dtype=np.float32)
            else:
                action = np.array([0, 0, 0, np.clip(insert_arm-arm, -0.1, 0.1), vac], dtype=np.float32)
        old_b = (bx, by)
        state, reward, term, trunc, info = env.step(action)
        nbx, nby = val(state, block, "x"), val(state, block, "y")
        new_vac = val(state, state.get_objects(typ.get_type("crv_robot"))[0], "vacuum")
        moved = math.hypot(nbx - old_b[0], nby - old_b[1]) > 1e-4
        was_holding = holding
        if not holding and new_vac > 0.5 and (moved or contact_ready):
            holding = True
            nr = state.get_objects(typ.get_type("crv_robot"))[0]
            grasp_offset = val(state, block, "theta") - val(state, nr, "theta")
            print(target_mode, "grasp inferred", step, "gripper", gx, gy, "block", old_b)
        new_target_dist = math.hypot(nbx - target[0], nby - target[1])
        ntheta = val(state, block, "theta")
        half_x = (abs(math.cos(ntheta))*val(state, block, "width") +
                  abs(math.sin(ntheta))*val(state, block, "height"))/2
        half_y = (abs(math.sin(ntheta))*val(state, block, "width") +
                  abs(math.cos(ntheta))*val(state, block, "height"))/2
        slot_x0 = val(state, shelf, "x1")
        slot_x1 = slot_x0 + val(state, shelf, "width1")
        slot_y0 = val(state, shelf, "y1") - val(state, shelf, "height1")/2
        slot_y1 = val(state, shelf, "y1") + val(state, shelf, "height1")/2
        contained = (nbx-half_x >= slot_x0 and nbx+half_x <= slot_x1 and
                     nby-half_y >= slot_y0 and nby+half_y <= slot_y1)
        if was_holding and not released and (new_target_dist < 0.035 or contained):
            # next iteration command vacuum off
            released = True
            print(target_mode, "release requested", step, "target", target, "block", (nbx, nby))
        if term or trunc:
            print(target_mode, "END", step + 1, "term", term, "trunc", trunc,
                  "block", (round(nbx, 4), round(nby, 4)), "reward", reward, "info", info)
            env.close()
            return
        if released and step % 10 == 0:
            print(target_mode, "post-release", step, "block", (round(nbx, 4), round(nby, 4)))
            if step > 400:
                break
    rr = state.get_objects(typ.get_type("crv_robot"))[0]
    print(target_mode, "NO END", "holding", holding, "released", released,
          "robot", tuple(round(val(state, rr, k), 4) for k in ("x", "y", "theta", "arm_joint")),
          "block", tuple(round(val(state, block, k), 4) for k in ("x", "y", "theta")))
    env.close()


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "summary":
        for probe_seed in range(4):
            summarize(probe_seed, 1)
    else:
        move_to_grasp_and_target(int(sys.argv[2]), sys.argv[1])
