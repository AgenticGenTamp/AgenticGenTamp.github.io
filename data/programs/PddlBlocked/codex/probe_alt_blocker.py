"""Probe PR2Blocked blocker geometry and alternative deterministic grasps."""
import math
import sys

import numpy as np

from env_client import make_env
from approach import GeneratedApproach


SEEDS = [11, 14, 21, 34, 42, 47]


def get(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def main():
    env = make_env()
    for seed in SEEDS if len(sys.argv) == 1 else map(int, sys.argv[1:]):
        state, _ = env.reset(seed=seed)
        green = np.array([get(state, "green0", "pose_x"), get(state, "green0", "pose_y")])
        blocker = np.array([get(state, "blocker", "pose_x"), get(state, "blocker", "pose_y")])
        direction = (blocker-green)/np.linalg.norm(blocker-green)
        print(seed, "green", green.round(5), "block", blocker.round(5),
              "d", direction.round(5), "base", [get(state, "robot", "base_x"), get(state, "robot", "base_y")])
    env.close()


def search(seed, trials=800):
    """Randomly search arm/base perturbations from the clipped nominal pose."""
    env = make_env()
    rng = np.random.default_rng(seed + 901)
    state, info = env.reset(seed=seed)
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    # Reach the closest feasible nominal pose (policy stalls against a base limit).
    for _ in range(60):
        state, *_ = env.step(policy.get_action(state))
    nominal_base = np.array([get(state, "robot", "base_x"), get(state, "robot", "base_y")])
    nominal_q = policy.Q.copy()
    for trial in range(trials):
        # Mostly preserve the known horizontal side-grasp wrist configuration while
        # bending/panning the shoulder and elbow to pull the tool inside base limits.
        scale = min(1.0, 0.25 + trial / 300.0)
        q = nominal_q + rng.normal(0, [0.65, 0.35, 0.35, 0.8, 0.5, 0.5, 0.5]) * scale
        q[:4] = np.clip(q[:4], [-2.28, -.52, -math.pi, -2.32], [.71, 1.39, math.pi, 0.0])
        q[4:] = (q[4:] + math.pi) % (2*math.pi) - math.pi
        target_base = nominal_base + rng.uniform([-.25, -.18], [.25, .18])
        target_base = np.clip(target_base, [-1., -1.], [5., 1.])
        for _ in range(18):
            action = np.zeros(11, np.float32)
            action[:2] = np.clip(target_base - np.array([get(state,"robot","base_x"), get(state,"robot","base_y")]), -.2, .2)
            for j in range(7):
                delta = q[j] - get(state, "robot", "joint_"+str(j+1))
                if j in (4, 6): delta = (delta+math.pi)%(2*math.pi)-math.pi
                action[3+j] = np.clip(delta, -.2, .2)
            action[10] = 1.
            state, *_ = env.step(action)
        action = np.zeros(11, np.float32); action[10] = -1.
        state, *_ = env.step(action)
        if get(state, "robot", "grasp_active") > .5:
            held = [name for name in ("blocker", "green0") if get(state,name,"grasp_active") > .5]
            print("HIT", seed, trial, "held", held, "base", [round(get(state,"robot",f),6) for f in ("base_x","base_y","base_rot")],
                  "q", [round(get(state,"robot","joint_"+str(j+1)),6) for j in range(7)],
                  "tf", [round(get(state,"robot","grasp_tf_"+f),6) for f in "xyz"])
            env.close(); return
        action[10] = 1.; state, *_ = env.step(action)
    print("MISS", seed)
    env.close()


def angle_search(seed):
    """Try the calibrated side-grasp posture from alternate world headings."""
    env = make_env()
    off = np.array([.94433737, .34963603])
    q = GeneratedApproach.Q
    probe, _ = env.reset(seed=seed)
    green0=np.array([get(probe,"green0","pose_x"),get(probe,"green0","pose_y")])
    block0=np.array([get(probe,"blocker","pose_x"),get(probe,"blocker","pose_y")])
    d=(block0-green0)/np.linalg.norm(block0-green0)
    nominal=(math.atan2(-d[1],-d[0])-.7758+math.pi)%(2*math.pi)-math.pi
    headings=np.r_[nominal, np.arange(-.8,.801,.01),
                   nominal+np.arange(.01,3.151,.03), nominal-np.arange(.01,3.151,.03)]
    for theta in headings:
        state, _ = env.reset(seed=seed)
        blocker = np.array([get(state,"blocker","pose_x"), get(state,"blocker","pose_y")])
        c, z = math.cos(theta), math.sin(theta)
        world_off = np.array([c*off[0]-z*off[1], z*off[0]+c*off[1]])
        target = blocker-world_off
        if np.any(target < [-1.,-1.]) or np.any(target > [5.,1.]): continue
        # Deploy in free space, route around the near table at |y|=1 when useful.
        side=.98 if target[1]>=0 else -.98
        waypoints = [target] if target[0]<=3.65 else [np.array([3.4,side]),np.array([target[0],side]),target]
        collided = False
        for wi, waypoint in enumerate(waypoints):
            for step in range(35):
                action=np.zeros(11,np.float32)
                xy=np.array([get(state,"robot","base_x"),get(state,"robot","base_y")])
                action[:2]=np.clip(waypoint-xy,-.2,.2)
                action[2]=np.clip((theta-get(state,"robot","base_rot")+math.pi)%(2*math.pi)-math.pi,-.2,.2)
                if wi == len(waypoints)-1:
                    for j in range(7):
                        delta=q[j]-get(state,"robot","joint_"+str(j+1))
                        if j in (4,6):delta=(delta+math.pi)%(2*math.pi)-math.pi
                        action[3+j]=np.clip(delta,-.2,.2)
                action[10]=1.; old=xy
                state,*_=env.step(action)
                xy=np.array([get(state,"robot","base_x"),get(state,"robot","base_y")])
                if np.max(np.abs(action[:2]))>.01 and np.max(np.abs(xy-old))<1e-5: collided=True
                arm_error=max(abs(((q[j]-get(state,"robot","joint_"+str(j+1))+math.pi)%(2*math.pi)-math.pi)
                                  if j in (4,6) else q[j]-get(state,"robot","joint_"+str(j+1))) for j in range(7))
                if (np.max(np.abs(xy-waypoint))<.004 and
                    abs((theta-get(state,"robot","base_rot")+math.pi)%(2*math.pi)-math.pi)<.004 and
                    (wi < len(waypoints)-1 or arm_error < .004)): break
        action=np.zeros(11,np.float32);action[10]=-1.;state,*_=env.step(action)
        if get(state,"blocker","grasp_active")>.5:
            print("ANGLE_HIT",seed,"theta",round(theta,7),"base",target.round(7).tolist(),
                  "q",q.tolist(),"collided",collided); env.close(); return
    print("ANGLE_MISS",seed);env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "search": search(int(sys.argv[2]))
    elif len(sys.argv)>1 and sys.argv[1] == "angles": angle_search(int(sys.argv[2]))
    else: main()
