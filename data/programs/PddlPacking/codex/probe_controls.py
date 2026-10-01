"""Empirical probes for PR2Packed action coordinates (never imported by policy)."""
import argparse
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from env_client import make_env


RF = [
    "base_x", "base_y", "base_rot", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7", "gripper_opening",
    "grasp_active",
]


def robot(state):
    obj = state.get_object_from_name("robot")
    return np.array([state.get(obj, f) for f in RF], dtype=float)


def objects(state):
    out = {}
    for name in state.get_object_names():
        if name != "robot":
            obj = state.get_object_from_name(name)
            out[name] = np.array([state.get(obj, "pose_x"),
                                  state.get(obj, "pose_y"),
                                  state.get(obj, "pose_z")])
    return out


def one_step(seed=0):
    print("ONE STEP, each coordinate from identical reset seed", seed)
    for i in range(11):
        for amount in ([0.2, -0.2] if i < 10 else [1.0, -1.0, 0.0]):
            env = make_env()
            s, _ = env.reset(seed=seed)
            before = robot(s)
            a = np.zeros(11, np.float32)
            a[i] = amount
            s, rew, term, trunc, _ = env.step(a)
            after = robot(s)
            changed = np.where(np.abs(after - before) > 1e-7)[0]
            print(i, amount, "changed", changed.tolist(),
                  "delta", np.round(after - before, 6).tolist(),
                  "reward/end", rew, term, trunc)
            env.close()


def limits(seed=0):
    print("LIMITS/WRAP, repeated coordinate-only actions")
    for i in range(10):
        results = []
        for amount in (0.2, -0.2):
            env = make_env()
            s, _ = env.reset(seed=seed)
            initial = robot(s)[i]
            vals = [initial]
            rejects = 0
            for _ in range(50):
                a = np.zeros(11, np.float32)
                a[i] = amount
                ns, _, _, _, _ = env.step(a)
                nv = robot(ns)[i]
                if abs(nv - vals[-1]) < 1e-8:
                    rejects += 1
                vals.append(nv)
                s = ns
            results.append((amount, initial, vals[-1], min(vals), max(vals), rejects,
                            np.round(vals[:8], 3).tolist(),
                            np.round(vals[-8:], 3).tolist()))
            env.close()
        print("coord", i, results)


def initial(seed=0):
    env = make_env()
    s, info = env.reset(seed=seed)
    print("seed", seed, "info", info, "robot", dict(zip(RF, robot(s))))
    print("objects", objects(s))
    env.close()


def grasp(seed=0):
    """Test a standard-PR2 FK hypothesis using block1 on seed 0."""
    env = make_env()
    s, _ = env.reset(seed=seed)
    blocks = [n for n in s.get_object_names() if n.startswith("block")]
    # Choose leftmost block to stay outside the table/base collision boundary.
    target_name = min(blocks, key=lambda n: s.get(s.get_object_from_name(n), "pose_x"))
    b = s.get_object_from_name(target_name)
    bx, by = s.get(b, "pose_x"), s.get(b, "pose_y")
    # FK predicts tool offset from base (0.652064, 0.250032, 0.774732)
    # for these joints; tool x-axis points vertically down.
    qtarget = np.array([0.72185326, 0.13229249, 1.56448917, -1.70117756,
                        1.70091685, -1.59432794, 2.16422769])
    target = np.r_[bx - 0.652064, by - 0.250032, 0.0, qtarget]
    print("grasp target", target_name, bx, by, "robot target", target)
    for stage, inds in (("base", range(3)), ("arm", range(3, 10))):
        for k in range(30):
            cur = robot(s)[:10]
            err = target - cur
            if max(abs(err[list(inds)])) < 2e-4:
                break
            a = np.zeros(11, np.float32)
            for i in inds:
                a[i] = np.clip(err[i], -.2, .2)
            ns, _, _, _, _ = env.step(a)
            if np.allclose(robot(ns)[:10], cur):
                print("whole motion rejected", stage, k, np.round(cur, 4),
                      "attempt", np.round(a, 4))
            s = ns
        print(stage, "reached", np.round(robot(s), 5))
    a = np.zeros(11, np.float32); a[10] = -1
    s, _, _, _, _ = env.step(a)
    print("after close", dict(zip(RF, robot(s))))
    print("block grasp flags", {n: s.get(s.get_object_from_name(n), "grasp_active")
                                for n in blocks})
    env.close()


def scan(seed=0):
    """Scan horizontal tool offset around the FK estimate at a fixed height."""
    env = make_env(); s, _ = env.reset(seed=seed)
    blocks = [n for n in s.get_object_names() if n.startswith("block")]
    target_name = min(blocks, key=lambda n: s.get(s.get_object_from_name(n), "pose_x"))
    b = s.get_object_from_name(target_name)
    bx, by = s.get(b, "pose_x"), s.get(b, "pose_y")
    qtarget = np.array([0.72185326, 0.13229249, 1.56448917, -1.70117756,
                        1.70091685, -1.59432794, 2.16422769])
    # First reach arm configuration while far enough from the table.
    for desired, inds in ((np.r_[-1.0, 0., 0., qtarget], range(3, 10)),):
        for _ in range(20):
            cur = robot(s)[:10]; a = np.zeros(11, np.float32)
            for i in inds: a[i] = np.clip(desired[i] - cur[i], -.2, .2)
            s, _, _, _, _ = env.step(a)
    nominal = np.array([bx - .652064, by - .250032])
    for dx in np.linspace(-.12, .12, 9):
        for dy in np.linspace(-.12, .12, 9):
            dest = nominal + [dx, dy]
            for _ in range(5):
                cur = robot(s); a = np.zeros(11, np.float32)
                a[:2] = np.clip(dest-cur[:2], -.2, .2)
                s, _, _, _, _ = env.step(a)
            a = np.zeros(11, np.float32); a[10] = -1
            s, _, _, _, _ = env.step(a)
            if robot(s)[11] > .5:
                print("SUCCESS offset", dx, dy, "base", robot(s)[:2], target_name)
                env.close(); return
            a[10] = 1; s, _, _, _, _ = env.step(a)
    print("no horizontal success")
    env.close()


def fk(q):
    p = np.zeros(3); rot = np.eye(3)
    axes = "zyxyxyx"
    # Link offsets occur before the following joint: shoulder .1, upper arm .4,
    # forearm .321, wrist/gripper .18.
    for qi, axis, length in zip(q, axes, (.1, 0., .4, .321, 0., .18, 0.)):
        rot = rot @ Rotation.from_rotvec(qi * np.eye(3)["xyz".index(axis)]).as_matrix()
        p += rot @ np.array([length, 0., 0.])
    return p, rot


def scan3d(seed=0):
    """Scan FK-derived vertical configurations and small horizontal errors."""
    env = make_env(); s, _ = env.reset(seed=seed)
    r0 = robot(s); q0 = r0[3:10].copy(); p0, rot0 = fk(q0)
    blocks = [n for n in s.get_object_names() if n.startswith("block")]
    target_name = min(blocks, key=lambda n: s.get(s.get_object_from_name(n), "pose_x"))
    b = s.get_object_from_name(target_name)
    bxy = np.array([s.get(b, "pose_x"), s.get(b, "pose_y")])
    lo = np.array([-.714602, -.5236, -.8, -2.3213, -20, -2.094, -20])
    hi = np.array([2.285398, 1.3963, 3.9, 0, 20, 0, 20])
    for down in np.linspace(.16, .70, 19):
        pt = p0 - [0, 0, down]
        def residual(q):
            p, rr = fk(q)
            return np.r_[10*(p-pt), Rotation.from_matrix(rot0.T @ rr).as_rotvec(),
                            .005*(q-q0)]
        q = least_squares(residual, q0, bounds=(lo, hi), max_nfev=1000).x
        p, _ = fk(q)
        # reach q outside table, then base alignment scan
        destbase = bxy - (np.array([.1, .188]) + p[:2])
        for _ in range(20):
            cur=robot(s); a=np.zeros(11,np.float32); a[:2]=np.clip([-1.,0.]-cur[:2],-.2,.2)
            a[3:10]=np.clip(q-cur[3:10],-.2,.2); s,*_=env.step(a)
        for dx in np.linspace(-.06,.06,5):
            for dy in np.linspace(-.06,.06,5):
                dest=destbase+[dx,dy]
                for _ in range(5):
                    cur=robot(s); a=np.zeros(11,np.float32); a[:2]=np.clip(dest-cur[:2],-.2,.2)
                    s,*_=env.step(a)
                a=np.zeros(11,np.float32);a[10]=-1;s,*_=env.step(a)
                if robot(s)[11]>.5:
                    print("SUCCESS down/offset",down,dx,dy,"q",q,"base",robot(s)[:2])
                    env.close();return
                a[10]=1;s,*_=env.step(a)
        print("tested down", down, "q", np.round(q,3), "wanted base", np.round(destbase,3),
              "actual", np.round(robot(s)[:10],3))
    print("no 3d success"); env.close()


def scanseq(seed=0):
    """Align above a block first, then descend (necessary near table)."""
    for down in np.linspace(.28, .40, 13):
        env=make_env();s,_=env.reset(seed=seed);r0=robot(s);q0=r0[3:10];p0,rot0=fk(q0)
        blocks=[n for n in s.get_object_names() if n.startswith("block")]
        n=min(blocks,key=lambda x:s.get(s.get_object_from_name(x),"pose_x"));b=s.get_object_from_name(n)
        bxy=np.array([s.get(b,"pose_x"),s.get(b,"pose_y")])
        lo=np.array([-.714602,-.5236,-.8,-2.3213,-20,-2.094,-20]);hi=np.array([2.285398,1.3963,3.9,0,20,0,20])
        def res(q):
            p,rr=fk(q);return np.r_[10*(p-(p0-[0,0,down])),Rotation.from_matrix(rot0.T@rr).as_rotvec(),.005*(q-q0)]
        q=least_squares(res,q0,bounds=(lo,hi),max_nfev=1000).x;p,_=fk(q)
        dest=bxy-(np.array([0.,.188])+p[:2])
        for _ in range(8):
            cur=robot(s);a=np.zeros(11,np.float32);a[:2]=np.clip(dest-cur[:2],-.2,.2);s,*_=env.step(a)
        for _ in range(20):
            cur=robot(s);a=np.zeros(11,np.float32);a[3:10]=np.clip(q-cur[3:10],-.1,.1);s,*_=env.step(a)
        a=np.zeros(11,np.float32);a[10]=-1;s,*_=env.step(a)
        print("seq",down,"wanted",np.round(dest,3),"actual",np.round(robot(s),3))
        if robot(s)[11]>.5: print("SUCCESS",down,q);env.close();return
        env.close()


def scan_descent(seed=0):
    down=.31
    for dx in np.linspace(-.08,.08,5):
      for dy in np.linspace(-.08,.08,5):
        env=make_env();s,_=env.reset(seed=seed);r0=robot(s);q0=r0[3:10];p0,rot0=fk(q0)
        ns=[n for n in s.get_object_names() if n.startswith("block")];n=min(ns,key=lambda x:s.get(s.get_object_from_name(x),"pose_x"));b=s.get_object_from_name(n)
        bxy=np.array([s.get(b,"pose_x"),s.get(b,"pose_y")]);lo=np.array([-.714602,-.5236,-.8,-2.3213,-20,-2.094,-20]);hi=np.array([2.285398,1.3963,3.9,0,20,0,20])
        def res(q):
          pp,rr=fk(q);return np.r_[10*(pp-(p0-[0,0,down])),Rotation.from_matrix(rot0.T@rr).as_rotvec(),.005*(q-q0)]
        q=least_squares(res,q0,bounds=(lo,hi),max_nfev=1000).x;p,_=fk(q);dest=bxy-(np.array([0,.188])+p[:2])+[dx,dy]
        for _ in range(8):
          cur=robot(s);a=np.zeros(11,np.float32);a[:2]=np.clip(dest-cur[:2],-.2,.2);s,*_=env.step(a)
        for _ in range(20):
          cur=robot(s);a=np.zeros(11,np.float32);a[3:10]=np.clip(q-cur[3:10],-.05,.05);s,*_=env.step(a)
        err=np.max(abs(robot(s)[3:10]-q));a=np.zeros(11,np.float32);a[10]=-1;s,*_=env.step(a)
        print("descent",dx,dy,"qerr",round(err,3),"grasp",robot(s)[11])
        if robot(s)[11]>.5: print("SUCCESS",dx,dy,q,dest);env.close();return
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initial", "one", "limits", "grasp", "scan", "scan3d", "scanseq", "scan_descent"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"initial": initial, "one": one_step, "limits": limits,
     "grasp": grasp, "scan": scan, "scan3d": scan3d, "scanseq": scanseq,
     "scan_descent": scan_descent}[args.mode](args.seed)
