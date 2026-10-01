"""Empirical probes for ClutteredStorage2D robot kinematics and grasping."""
import argparse
import math
import numpy as np

from env_client import make_env


def values(state):
    out = {}
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        out[name] = {f: float(state.get(obj, f)) for f in state.type_features[obj.type]}
    return out


def tip(d):
    r = d["robot"]
    # Candidate center-line gripper center based on arm geometry.
    # Empirically measured vacuum/contact point (approximately joint + 0.307).
    reach = r["arm_joint"] + 0.307
    return (r["x"] + reach * math.cos(r["theta"]),
            r["y"] + reach * math.sin(r["theta"]))


def summarize(d):
    r = d["robot"]
    print("robot", {k: round(r[k], 5) for k in
                    ("x", "y", "theta", "base_radius", "arm_joint",
                     "arm_length", "vacuum", "gripper_height", "gripper_width")})
    print("tip_candidate", tuple(round(x, 5) for x in tip(d)))
    for n in sorted(k for k in d if k.startswith("block")):
        b = d[n]
        print(n, {k: round(b[k], 5) for k in ("x", "y", "theta", "width", "height")})
    s = d["shelf"]
    print("shelf", {k: round(s[k], 5) for k in
                    ("x", "y", "theta", "width", "height", "x1", "y1", "theta1", "width1", "height1")})


def initial(seed):
    env = make_env()
    s, _ = env.reset(seed=seed)
    summarize(values(s))
    env.close()


def motion(seed):
    env = make_env()
    s, _ = env.reset(seed=seed)
    old = values(s)
    summarize(old)
    actions = [
        [0.05, 0, 0, 0, 0], [0, 0.05, 0, 0, 0],
        [0, 0, math.pi / 16, 0, 0], [0, 0, 0, 0.1, 0],
        [0, 0, -math.pi / 16, -0.1, 1],
    ]
    for a in actions:
        s, rew, term, trunc, _ = env.step(np.array(a, dtype=np.float32))
        new = values(s)
        r0, r1 = old["robot"], new["robot"]
        print("action", a, "delta_robot",
              {k: round(r1[k] - r0[k], 6) for k in ("x", "y", "theta", "arm_joint", "arm_length", "vacuum")},
              "reward", rew, "done", term, trunc)
        old = new
    env.close()


def joint_bounds(seed):
    env = make_env(); s, _ = env.reset(seed=seed)
    for sign in (-1, 1):
        for i in range(12):
            s,_,_,_,_=env.step(np.array([0,0,0,sign*0.1,0],dtype=np.float32))
            print(sign, i, values(s)["robot"]["arm_joint"])
    env.close()


def drive_delta(env, s, dx, dy, dtheta, joint_target, vacuum):
    """Drive independent directly-controlled coordinates to a relative target."""
    goal = values(s)["robot"]
    gx, gy = goal["x"] + dx, goal["y"] + dy
    gt = goal["theta"] + dtheta
    for _ in range(200):
        r = values(s)["robot"]
        ex, ey = gx-r["x"], gy-r["y"]
        et = (gt-r["theta"]+math.pi)%(2*math.pi)-math.pi
        ej = joint_target-r["arm_joint"]
        if max(abs(ex),abs(ey),abs(et),abs(ej)) < 1e-5: break
        a=[max(-.05,min(.05,ex)),max(-.05,min(.05,ey)),max(-math.pi/16,min(math.pi/16,et)),max(-.1,min(.1,ej)),vacuum]
        s,*_=env.step(np.array(a,dtype=np.float32))
    return s


def scan_reach(seed, block_name="block3"):
    for joint in (.2,.3,.4,.5):
      for dist in np.arange(.30, 1.51, .05):
        env=make_env(); s,_=env.reset(seed=seed); d=values(s); r=d["robot"]; b=d[block_name]
        goalx,goaly=b["x"]-dist,b["y"]
        # Position and orient while vacuum off. Compare block to catch pushing.
        s=drive_delta(env,s,goalx-r["x"],goaly-r["y"],-r["theta"],joint,0)
        q=values(s); before=(q[block_name]["x"],q[block_name]["y"])
        # Vacuum on, then retract whole base away from block (cannot push it).
        s,*_=env.step(np.array([0,0,0,0,1],dtype=np.float32))
        s,*_=env.step(np.array([-.05,0,0,0,1],dtype=np.float32))
        q=values(s); after=(q[block_name]["x"],q[block_name]["y"])
        delta=math.hypot(after[0]-before[0],after[1]-before[1])
        if delta>1e-5:
            print("GRIP/PUSH", "joint",joint,"dist",round(float(dist),3),"delta",round(delta,5),"before",before,"after",after)
        env.close()


def threshold(seed, block_name="block3"):
    """Fine radial and lateral grasp envelope around joint=.5, theta=0."""
    for axis in ("radial", "lateral"):
      hits=[]
      step = .002 if axis == "radial" else .01
      span = .06 if axis == "radial" else .20
      for off in np.arange(-span,span+.0001,step):
        env=make_env(); s,_=env.reset(seed=seed); d=values(s); r=d["robot"]; b=d[block_name]
        dist=.8 + (off if axis=="radial" else 0)
        yoff=off if axis=="lateral" else 0
        gx,gy=b["x"]-dist,b["y"]+yoff
        s=drive_delta(env,s,gx-r["x"],gy-r["y"],-r["theta"],.5,0)
        q=values(s); before=(q[block_name]["x"],q[block_name]["y"])
        s,*_=env.step(np.array([0,0,0,0,1],dtype=np.float32))
        s,*_=env.step(np.array([-.05,0,0,0,1],dtype=np.float32))
        q=values(s); after=(q[block_name]["x"],q[block_name]["y"])
        if math.hypot(after[0]-before[0],after[1]-before[1])>.04: hits.append(round(float(off),3))
        env.close()
      print(axis,"hits",hits)


def carry(seed, block_name="block3"):
    env=make_env(); s,_=env.reset(seed=seed); d=values(s); r=d["robot"]; b=d[block_name]
    gx,gy=b["x"]-.8,b["y"]
    s=drive_delta(env,s,gx-r["x"],gy-r["y"],-r["theta"],.5,0)
    acts=([0,0,0,0,1],[-.05,0,0,0,1],[0,.05,0,0,1],[0,0,.1,0,1],
          [0,0,0,-.1,1],[0,0,0,0,0],[0,.05,0,0,0])
    for a in acts:
      old=values(s); s,*_=env.step(np.array(a,dtype=np.float32)); q=values(s)
      print("action",a,"robot",[round(q['robot'][k],5) for k in ('x','y','theta','arm_joint','vacuum')],
            "block",[round(q[block_name][k],5) for k in ('x','y','theta')],
            "delta",[round(q[block_name][k]-old[block_name][k],5) for k in ('x','y','theta')])
    env.close()


def scripted_contact(seed, block_name="block0"):
    """Use closed-loop base/angle/arm commands to put candidate tip on a block."""
    env = make_env()
    s, _ = env.reset(seed=seed)
    d = values(s)
    summarize(d)
    carried = False
    for t in range(250):
        d = values(s); r = d["robot"]; b = d[block_name]
        vx, vy = b["x"] - r["x"], b["y"] - r["y"]
        angle = math.atan2(vy, vx)
        da = (angle - r["theta"] + math.pi) % (2 * math.pi) - math.pi
        dist = math.hypot(vx, vy)
        desired_arm = dist - 0.307
        # Rotate/retract first, then translate base if target outside arm range.
        if abs(da) > 0.005:
            a = [0, 0, max(-math.pi/16, min(math.pi/16, da)), -0.1, 0]
        elif desired_arm > 0.49:
            move = min(0.05, desired_arm - 0.45)
            a = [move * math.cos(angle), move * math.sin(angle), 0, 0, 0]
        elif abs(desired_arm-r["arm_joint"]) > 0.002:
            a = [0, 0, 0, max(-0.1, min(0.1, desired_arm-r["arm_joint"])), 0]
        else:
            a = [0, 0, 0, 0, 1]
        oldb = (b["x"], b["y"], b["theta"])
        s, rew, term, trunc, _ = env.step(np.array(a, dtype=np.float32))
        nd = values(s); nr = nd["robot"]; nb = nd[block_name]
        moveb = math.hypot(nb["x"]-oldb[0], nb["y"]-oldb[1])
        if t % 10 == 0 or moveb > 1e-5 or nr["vacuum"] != r["vacuum"]:
            tx, ty = tip(nd)
            print(t, "a", [round(x,3) for x in a], "r", [round(nr[x],3) for x in ("x","y","theta","arm_length","vacuum")],
                  "tip", [round(tx,3),round(ty,3)], "b", [round(nb[x],3) for x in ("x","y","theta")], "bmove", round(moveb,5))
        if moveb > 1e-4:
            carried = True
        if carried and t > 5:
            # Continue arbitrary motions to identify carried transform, then release.
            for aa in ([0.05,0,0,0,1], [0,0,math.pi/16,0,1], [0,0,0,-0.1,1], [0,0,0,0,0]):
                prev=values(s); s,_,_,_,_=env.step(np.array(aa,dtype=np.float32)); cur=values(s)
                print("carry action",aa,"robot",[round(cur['robot'][x],4) for x in ('x','y','theta','arm_length','vacuum')],
                      "block",[round(cur[block_name][x],4) for x in ('x','y','theta')],
                      "delta_block",[round(cur[block_name][x]-prev[block_name][x],4) for x in ('x','y','theta')])
            break
        if term or trunc: break
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("initial","motion","bounds","scan","threshold","carry","contact")); p.add_argument("--seed", type=int, default=0); p.add_argument("--block",default="block0")
    a=p.parse_args()
    {"initial": initial, "motion": motion, "bounds": joint_bounds, "scan": scan_reach, "threshold": threshold, "carry": carry, "contact": scripted_contact}[a.mode](a.seed, **({"block_name":a.block} if a.mode in ("scan","threshold","carry","contact") else {}))
