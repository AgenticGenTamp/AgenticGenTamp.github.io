"""Focused black-box probes for BalanceBeam3D controls."""
import argparse
import json
import numpy as np

from env_client import make_env


def summary(s):
    return {
        "large": np.round(s[0:3], 3).tolist(),
        "base": np.round(s[16:19], 3).tolist(),
        "joints": np.round(s[19:27], 3).tolist(),
        "seesaw": np.round(s[38:41], 3).tolist(),
        "small1": np.round(s[54:57], 3).tolist(),
        "small2": np.round(s[70:73], 3).tolist(),
    }


def rollout(seed, action, steps):
    env = make_env()
    s, info = env.reset(seed=seed)
    start = s.copy()
    total = 0.0
    done = False
    rewards = []
    for t in range(steps):
        s, r, term, trunc, info = env.step(action.copy())
        total += r
        if r != -0.01:
            rewards.append((t + 1, round(float(r), 4)))
        if term or trunc:
            done = True
            break
    env.close()
    return start, s, total, done, rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["map", "grip", "sequence", "graspbase", "armcontact", "dump", "grasptry"], default="map")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--raw", action="store_true")
    args = p.parse_args()

    if args.mode == "map":
        z = np.zeros(11, np.float32)
        for idx in (int(x) for x in args.indices.split(",")):
            for sign in (1.0, -1.0):
                a = z.copy(); a[idx] = 0.1 * sign; a[10] = 1.0
                s0, s1, ret, done, rs = rollout(args.seed, a, args.steps)
                delta = np.round(s1[16:27] - s0[16:27], 3)
                print(idx, int(sign), "drobot", delta.tolist(), "obj", summary(s1), "R", round(ret,3), rs)
    elif args.mode == "grip":
        for g in (0.0, 1.0):
            a = np.zeros(11, np.float32); a[10] = g
            s0, s1, ret, done, rs = rollout(args.seed, a, 15)
            print("g", g, "start", summary(s0), "end", summary(s1), "R", ret, rs)
    elif args.mode == "graspbase":
        for g in (0.0, 1.0):
            env=make_env(); s,_=env.reset(seed=args.seed); start=s.copy()
            hist=[]
            phases=[((.075,-.04),6),((0.,0.),4),((-.075,.04),6)]
            for (ax,ay),n in phases:
                a=np.zeros(11,np.float32); a[0]=ax; a[1]=ay; a[10]=g
                for _ in range(n):
                    s,r,te,tr,info=env.step(a)
                hist.append((summary(s),round(float(r),3)))
            print("g",g,"start",summary(start),"phases",hist)
            env.close()
    elif args.mode == "armcontact":
        for idx in (4,6,8):
            for sign in (-1.,1.):
                env=make_env(); s,_=env.reset(seed=args.seed); start=s.copy()
                # Put the mobile base close to the large cube.
                a=np.zeros(11,np.float32); a[0]=.075; a[1]=-.04; a[10]=0
                for _ in range(6): s,r,te,tr,info=env.step(a)
                near=s.copy()
                a[:]=0; a[idx]=.1*sign; a[10]=0
                for _ in range(10): s,r,te,tr,info=env.step(a)
                print("joint",idx,"sign",int(sign),"near",summary(near),"end",summary(s),
                      "object_delta",np.round(np.r_[s[0:3]-start[0:3],s[54:57]-start[54:57],s[70:73]-start[70:73]],3).tolist())
                env.close()
    elif args.mode == "dump":
        env=make_env(); s,_=env.reset(seed=args.seed)
        # Optional comma list is interpreted as action-index:sign[:steps] here.
        for spec in args.indices.split(","):
            fields=[int(x) for x in spec.split(":")]
            idx,sign=fields[:2]; n=fields[2] if len(fields)>2 else args.steps
            a=np.zeros(11,np.float32); a[idx]=.1*sign; a[10]=0
            for _ in range(n): s,r,te,tr,info=env.step(a)
        print(json.dumps(s.tolist()))
        env.close()
    elif args.mode == "grasptry":
        env=make_env(); s,_=env.reset(seed=args.seed); start=s.copy()
        phases=[]
        def drive(vals,n):
            nonlocal s
            a=np.zeros(11,np.float32); a[10]=vals.get(10,0)
            for i,v in vals.items(): a[i]=v
            for _ in range(n): s,r,te,tr,info=env.step(a)
            phases.append(summary(s))
        drive({0:.075,1:.075},3)       # align near the two small cubes
        drive({4:.1},40)              # shoulder down
        drive({6:.1},40)              # elbow extends, fingers downward
        drive({4:.1},40)              # continue shoulder until fingers reach ground
        drive({10:1.},6)              # close
        drive({4:-.1,10:1.},20)       # lift/retract while closed
        print("start",summary(start),"phases",phases,
              "delta",np.round(np.r_[s[0:3]-start[0:3],s[54:57]-start[54:57],s[70:73]-start[70:73]],3).tolist())
        if args.raw: print(json.dumps(s.tolist()))
        env.close()
    else:
        env = make_env(); s, _ = env.reset(seed=args.seed)
        print("start", summary(s))
        # Exercise one control at a time while printing sparse configurations.
        for idx, sign, n in [(0,1,10),(1,1,10),(2,1,10),(3,1,10),(4,1,10),(5,1,10),(6,1,10),(7,1,10),(8,1,10),(9,1,10)]:
            a=np.zeros(11,np.float32); a[idx]=0.1*sign; a[10]=1
            for _ in range(n): s,r,te,tr,info=env.step(a)
            print("after",idx,summary(s),"r",r)
        env.close()


if __name__ == "__main__":
    main()
