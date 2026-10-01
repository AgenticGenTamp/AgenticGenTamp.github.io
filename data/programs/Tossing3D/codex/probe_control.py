import sys
import numpy as np
from env_client import make_env


def snap(state):
    out = {}
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        vals = {}
        for feature in (
            "x", "y", "z", "bb_x", "bb_y", "bb_z",
            "pos_base_x", "pos_base_y", "pos_base_rot",
            "pos_arm_joint1", "pos_arm_joint2", "pos_arm_joint3",
            "pos_arm_joint4", "pos_arm_joint5", "pos_arm_joint6",
            "pos_arm_joint7", "pos_gripper",
        ):
            try:
                vals[feature] = float(state.get(obj, feature))
            except (KeyError, ValueError):
                pass
        out[name] = vals
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = make_env()
    state, info = env.reset(seed=seed)
    print("max", env.max_steps, "names", state.get_object_names(), "info", info)
    before = snap(state)
    for k, v in before.items():
        print(k, v)

    # Individual one-step action channel response, resetting each time.
    for idx in range(18):
        state, _ = env.reset(seed=seed)
        a = np.zeros(18, dtype=np.float32)
        a[idx] = 0.05 if idx < 10 else (1.0 if idx == 10 else 0.5)
        state2, rew, term, trunc, info = env.step(a)
        b, c = snap(state), snap(state2)
        rname = next(n for n in c if n == "robot")
        changed = {k: round(c[rname][k] - b[rname][k], 5)
                   for k in c[rname] if abs(c[rname][k] - b[rname][k]) > 1e-5}
        print("channel", idx, "changed", changed, "reward", rew)
    env.close()


def drive():
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    env = make_env()
    s, info = env.reset(seed=seed)
    cube = s.get_object_from_name("cube_0")
    robot = s.get_object_from_name("robot")
    cx, cy = (float(s.get(cube, f)) for f in ("x", "y"))
    bx, by = (float(s.get(robot, f)) for f in ("pos_base_x", "pos_base_y"))
    print("drive start", bx, by, "cube", cx, cy)
    for t in range(80):
        a = np.zeros(18, dtype=np.float32)
        # First align y, then advance x through cube.
        if t < 10:
            a[1] = np.clip(cy - float(s.get(robot, "pos_base_y")), -.1, .1)
        else:
            a[0] = .1
        s, r, term, trunc, inf = env.step(a)
        if t % 5 == 4:
            print(t + 1, "base", *(round(float(s.get(robot, f)), 3) for f in ("pos_base_x", "pos_base_y")),
                  "cube", *(round(float(s.get(cube, f)), 3) for f in ("x", "y", "z")), "r", r)
        if term or trunc:
            break
    env.close()


def grasp_grid():
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for dx, dy in ((.35, 0.), (.45, 0.), (.55, 0.), (.65, 0.)):
            env = make_env()
            s, _ = env.reset(seed=seed)
            cube = s.get_object_from_name("cube_0")
            robot = s.get_object_from_name("robot")
            cx0, cy0, cz0 = (float(s.get(cube, f)) for f in ("x", "y", "z"))
            tx, ty = cx0 - dx, cy0 - dy
            for t in range(15):
                a = np.zeros(18, dtype=np.float32)
                a[0] = np.clip(tx - float(s.get(robot, "pos_base_x")), -.1, .1)
                a[1] = np.clip(ty - float(s.get(robot, "pos_base_y")), -.1, .1)
                s, *_ = env.step(a)
            for t in range(10):
                a = np.zeros(18, dtype=np.float32); a[10] = 1.
                s, *_ = env.step(a)
            # Pull laterally. If grasped, cube should follow.
            for t in range(8):
                a = np.zeros(18, dtype=np.float32); a[1] = -.08; a[10] = 1.
                s, *_ = env.step(a)
            cx, cy, cz = (float(s.get(cube, f)) for f in ("x", "y", "z"))
            print("offset", dx, dy, "delta cube", round(cx-cx0,3), round(cy-cy0,3), round(cz-cz0,3),
                  "base", round(float(s.get(robot,"pos_base_x")),2), round(float(s.get(robot,"pos_base_y")),2))
            env.close()


def joint_response():
    for mode in ("pos", "vel", "both"):
        env = make_env(); s, _ = env.reset(seed=0)
        robot = s.get_object_from_name("robot")
        q0 = float(s.get(robot, "pos_arm_joint2"))
        for t in range(10):
            a = np.zeros(18, dtype=np.float32)
            if mode in ("pos", "both"): a[4] = .1
            if mode in ("vel", "both"): a[12] = 1.
            s, *_ = env.step(a)
            print(mode, t+1, round(float(s.get(robot,"pos_arm_joint2"))-q0, 4))
        env.close()


def lowered_pose():
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    target = np.array([0., 0., np.pi, -1.40, 0., -1.76, np.pi/2])
    for close_value in (1., 0.):
      for dx in (.3, .4, .5, .6):
        env=make_env(); s,_=env.reset(seed=seed)
        cube=s.get_object_from_name("cube_0"); robot=s.get_object_from_name("robot")
        start=np.array([float(s.get(cube,f)) for f in ("x","y","z")])
        tx=start[0]-dx; ty=start[1]
        # Approach base and joint pose with gripper opposite its closing value.
        for t in range(25):
            q=np.array([float(s.get(robot,"pos_arm_joint%d"%(j+1))) for j in range(7)])
            a=np.zeros(18,np.float32)
            a[0]=np.clip(tx-float(s.get(robot,"pos_base_x")),-.1,.1)
            a[1]=np.clip(ty-float(s.get(robot,"pos_base_y")),-.1,.1)
            a[10]=1.-close_value
            a[11:18]=np.clip(5*(target-q),-5,5)
            s,*_=env.step(a)
        atpose=np.array([float(s.get(cube,f)) for f in ("x","y","z")])
        # Close and settle.
        for t in range(8):
            a=np.zeros(18,np.float32); a[10]=close_value
            s,*_=env.step(a)
        # Pull laterally by 0.4 m while maintaining closure.
        for t in range(7):
            a=np.zeros(18,np.float32); a[1]=-.08; a[10]=close_value
            s,*_=env.step(a)
        end=np.array([float(s.get(cube,f)) for f in ("x","y","z")])
        print("close",close_value,"dx",dx,"posemove",np.round(atpose-start,3),"pullmove",np.round(end-atpose,3),
              "qerr",round(float(np.max(np.abs(q-target))),3),flush=True)
        env.close()


def elbow_sweep():
    for j4 in (-.5, -1., -1.5, -2., -2.5):
        env=make_env(); s,_=env.reset(seed=0)
        cube=s.get_object_from_name("cube_0"); robot=s.get_object_from_name("robot")
        start=np.array([float(s.get(cube,f)) for f in ("x","y","z")])
        target=np.array([0.,0.,np.pi,j4,0.,-np.pi-j4,np.pi/2])
        tx=start[0]-.7; ty=start[1]
        for t in range(25):
            q=np.array([float(s.get(robot,"pos_arm_joint%d"%(j+1))) for j in range(7)])
            a=np.zeros(18,np.float32); a[0]=np.clip(tx-float(s.get(robot,"pos_base_x")),-.1,.1); a[1]=np.clip(ty-float(s.get(robot,"pos_base_y")),-.1,.1)
            a[10]=1.; a[11:]=np.clip(5*(target-q),-5,5); s,*_=env.step(a)
        maxmove=0
        for t in range(12):
            a=np.zeros(18,np.float32); a[0]=.06; a[10]=1.; s,*_=env.step(a)
            p=np.array([float(s.get(cube,f)) for f in ("x","y","z")]); maxmove=max(maxmove,float(np.linalg.norm(p-start)))
        print("j4",j4,"move",round(maxmove,4),"base_x",round(float(s.get(robot,"pos_base_x")),3),flush=True)
        env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "drive":
        drive()
    elif len(sys.argv) > 1 and sys.argv[1] == "grid":
        grasp_grid()
    elif len(sys.argv) > 1 and sys.argv[1] == "joint":
        joint_response()
    elif len(sys.argv) > 1 and sys.argv[1] == "lower":
        lowered_pose()
    elif len(sys.argv) > 1 and sys.argv[1] == "elbow":
        elbow_sweep()
    else:
        main()
