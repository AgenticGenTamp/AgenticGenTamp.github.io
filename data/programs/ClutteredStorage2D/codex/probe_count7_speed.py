"""Benchmark and probe object ordering on fixed seven-block episodes."""

import sys
import math
from collections import Counter

import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def run(seed, mode):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 7})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    initial_stored = list(policy.to_clear)
    if mode == "leave_stored":
        policy.to_clear = []
    if mode == "no_direct":
        policy.direct_mode = False
    counts = Counter()
    order = []
    last = None
    term = trunc = False
    fast_phase = None
    shifted = False
    separate_phase = None
    separate_x = None
    push_ticks = 0
    for step in range(1500):
        repack_direct = (mode == "fast_repack" and policy.stage == "choose"
                         and policy.priority_name in initial_stored)
        if mode in ("robust4", "robust4_direct5", "robust4_deep", "cluster_first",
                    "cluster_after5"):
            policy.attempts["block4"] = max(1, policy.attempts.get("block4", 0))
        if mode == "fast_repack":
            policy.attempts["block4"] = max(1, policy.attempts.get("block4", 0))
        if (mode == "cluster_first" and policy.stage == "choose" and not policy.to_clear
                and all(policy._inside(state, state.get_object_from_name(n))
                        for n in initial_stored)):
            for cluster_name in ("block4", "block6"):
                cluster_block = state.get_object_from_name(cluster_name)
                if not policy._inside(state, cluster_block):
                    policy.priority_name = cluster_name
                    break
        if (mode == "cluster_after5" and policy.stage == "choose" and not policy.to_clear
                and policy._inside(state, state.get_object_from_name("block5"))):
            for cluster_name in ("block4", "block6"):
                cluster_block = state.get_object_from_name(cluster_name)
                if not policy._inside(state, cluster_block):
                    policy.priority_name = cluster_name
                    break
        if mode == "robust4_direct5" and policy.stage == "choose":
            policy.attempts["block5"] = 0
        if (mode == "push_pair" and policy.stage == "insert"
                and policy.target_name == "block4" and policy.insert_stall >= 5
                and push_ticks == 0):
            push_ticks = 1
        if (mode == "shift4" and policy.stage == "insert"
                and policy.target_name == "block4" and policy.insert_stall >= 5
                and not shifted):
            policy.place = (policy.place[0] - .16, policy.place[1])
            policy.preplace = (policy.place[0], policy.preplace[1])
            shifted = True
        if (mode == "pull_shift4" and policy.stage == "insert"
                and policy.target_name == "block4" and policy.insert_stall >= 5
                and separate_phase is None and not shifted):
            separate_phase = "pull"
            separate_x = policy.place[0] - .16
        if (mode == "priority6" and policy.stage == "choose" and not policy.to_clear
                and all(policy._inside(state, state.get_object_from_name(n))
                        for n in initial_stored)):
            b6 = state.get_object_from_name("block6")
            if not policy._inside(state, b6):
                policy.priority_name = "block6"
        forced_action = None
        if push_ticks:
            robot_now = state.get_object_from_name("robot")
            if push_ticks > 1:
                b6_now = state.get_object_from_name("block6")
                if policy._inside(state, b6_now):
                    push_ticks = -1
            if push_ticks == 1:
                forced_action = np.array([0, 0, 0, 0, 0], np.float32)
                push_ticks += 1
            elif push_ticks > 1:
                forced_action = np.array([0, .02, .05 if push_ticks % 2 else -.05,
                                          0, 0], np.float32)
                push_ticks += 1
            elif push_ticks == -1 and float(state.get(robot_now, "arm_joint")) > .21:
                forced_action = np.array([0, 0, 0, -.1, 0], np.float32)
            elif push_ticks == -1:
                push_ticks = -2
                forced_action = np.array([0, -.05, 0, 0, 0], np.float32)
            elif float(state.get(robot_now, "y")) > 1.70:
                forced_action = np.array([0, -.05, 0, 0, 0], np.float32)
            else:
                push_ticks = 0
                policy.stage = "choose"
                policy.target_name = None
        if separate_phase is not None:
            separated = state.get_object_from_name("block4")
            bx4 = float(state.get(separated, "x")); by4 = float(state.get(separated, "y"))
            if separate_phase == "pull" and by4 > 2.30:
                forced_action = np.array([0, -.03, 0, 0, 1], np.float32)
            elif separate_phase == "pull":
                separate_phase = "shift"
                forced_action = np.array([-.05, 0, 0, 0, 1], np.float32)
            elif separate_phase == "shift" and bx4 > separate_x + .01:
                forced_action = np.array([-.05, 0, 0, 0, 1], np.float32)
            else:
                policy.place = (separate_x, policy.place[1])
                policy.last_insert_y = by4
                policy.insert_stall = 0
                separate_phase = None
                shifted = True
        if mode in ("deep_clear", "robust4_deep") and policy.stage == "clear_pull":
            clear_target = state.get_object_from_name(policy.target_name)
            if float(state.get(clear_target, "y")) > 1.95:
                forced_action = np.array([0, -.03, 0, 0, 1], np.float32)
        if mode == "continuous_clear" and fast_phase is None and policy.stage == "clear_pull":
            target = state.get_object_from_name(policy.target_name)
            if float(state.get(target, "y")) <= 2.25:
                fast_phase = "lower"
        if forced_action is not None:
            action = forced_action
        if repack_direct and policy.target_name:
            repacked = state.get_object_from_name(policy.target_name)
            bxr = float(state.get(repacked, "x")); byr = float(state.get(repacked, "y"))
            btr = float(state.get(repacked, "theta"))
            policy.direct_pick = True
            policy.approach_theta = btr
            policy.goal_base = (bxr - .54 * math.cos(btr), byr - .54 * math.sin(btr))
            policy.stage = "navigate"
        elif fast_phase is not None:
            target = state.get_object_from_name(policy.target_name)
            robot = state.get_object_from_name("robot")
            by = float(state.get(target, "y")); bt = float(state.get(target, "theta"))
            arm = float(state.get(robot, "arm_joint"))
            if fast_phase == "lower" and by > 1.72:
                action = np.array([0, -.05, 0, 0, 1], np.float32)
            elif fast_phase == "lower":
                fast_phase = "retract"
                action = np.array([0, 0, 0, np.clip(.2-arm, -.1, .1), 1], np.float32)
            elif fast_phase == "retract" and arm > .21:
                action = np.array([0, 0, 0, np.clip(.2-arm, -.1, .1), 1], np.float32)
            elif fast_phase == "retract":
                fast_phase = "rotate"
                action = np.array([0, 0, np.clip((math.pi/2-bt+math.pi) %
                                                  (2*math.pi)-math.pi, -.196, .196), 0, 1],
                                  np.float32)
            elif abs((math.pi/2-bt+math.pi) % (2*math.pi)-math.pi) > .025:
                action = np.array([0, 0, np.clip((math.pi/2-bt+math.pi) %
                                                  (2*math.pi)-math.pi, -.196, .196), 0, 1],
                                  np.float32)
            else:
                policy.to_clear = [n for n in policy.to_clear if n != policy.target_name]
                policy.clear_mode = False
                policy.regrasped = True
                policy.stage = "lower_transport"
                fast_phase = None
                action = np.array([0, 0, 0, 0, 1], np.float32)
        else:
            if mode == "angle_gate" and policy.stage == "choose" and not policy.to_clear:
                robot = state.get_object_from_name("robot")
                rx = float(state.get(robot, "x")); ry = float(state.get(robot, "y"))
                outside = [b for b in state.get_objects(policy.block_type)
                           if not policy._inside(state, b)]
                if outside:
                    chosen = min(outside, key=lambda b:
                                 (float(state.get(b, "x"))-rx)**2 +
                                 (float(state.get(b, "y"))-ry)**2)
                    theta = (float(state.get(chosen, "theta"))+math.pi) % (2*math.pi)-math.pi
                    policy.direct_mode = abs(theta) <= 2.5
            action = policy.get_action(state)
        if mode == "batch_clear" and policy.priority_name is not None and policy.to_clear:
            policy.priority_name = None
        if mode == "outside_first" and policy.stage == "choose":
            policy.to_clear = []
        counts[policy.stage] += 1
        if policy.target_name != last:
            order.append((step, policy.target_name, policy.stage))
            last = policy.target_name
        state, reward, term, trunc, step_info = env.step(action)
        if term or trunc:
            break
    blocks = list(state.get_objects(policy.block_type))
    shelf = state.get_objects(policy.shelf_type)[0]
    sx = float(state.get(shelf, "x1")); sy = float(state.get(shelf, "y1"))
    sw = float(state.get(shelf, "width1")); sh = float(state.get(shelf, "height1"))
    inside = sum(sx <= float(state.get(b, "x")) <= sx+sw and
                 sy <= float(state.get(b, "y")) <= sy+sh for b in blocks)
    print(seed, mode, "steps", step+1, "OK" if term else "FAIL", "inside", inside,
          "order", order, "top", counts.most_common(6), flush=True)
    if not term:
        robot = state.get_object_from_name("robot")
        target = (state.get_object_from_name(policy.target_name)
                  if policy.target_name is not None else robot)
        print(" detail", policy.stage,
              tuple(round(float(state.get(robot, f)), 3) for f in
                    ("x", "y", "theta", "arm_joint", "vacuum")),
              tuple(round(float(state.get(target, f)), 3) for f in ("x", "y", "theta")))
        print(" blocks", [(b.name,) + tuple(round(float(state.get(b, f)), 3)
                                               for f in ("x", "y", "theta"))
                           for b in blocks])
    env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    seeds = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else list(range(5))
    for seed in seeds:
        run(seed, mode)
