import math
import numpy as np
from env_client import make_env


def objects(state, space):
    out = {}
    for typ in space.types:
        fs = space.type_features[typ]
        for obj in state.get_objects(typ):
            out[obj.name] = {f: float(state.get(obj, f)) for f in fs}
    return out


def act(r0=(0, 0, 0, 0), r1=(0, 0, 0, 0)):
    return np.asarray((*r0, *r1), dtype=np.float32)


def main(seed=0):
    env = make_env()
    s, info = env.reset(seed=seed)
    d = objects(s, env.observation_space)
    print("seed", seed, "info", info)
    for n, v in d.items(): print(n, v)
    print("motion probe rover0")
    for a in [(0.2, 0, 0, 0), (0, 0.2, 0, 0), (0, 0, 0.4, 0),
              (0.2, 0.2, 0.4, 0), (-0.2, -0.2, -0.4, 0)]:
        old = objects(s, env.observation_space)["rover0"]
        s, r, term, trunc, info = env.step(act(a))
        new = objects(s, env.observation_space)["rover0"]
        print(a, "old", old, "new", new, "r", r, term, trunc, info)
    env.close()


def collision_probe(seed=0):
    env = make_env(); s, _ = env.reset(seed=seed)
    sp = env.observation_space
    # Head east to the arena boundary in small increments.
    for direction in [(0.2, 0), (0, 0.2), (-0.2, 0), (0, -0.2)]:
        print("DIR", direction)
        for i in range(30):
            old = objects(s, sp)["rover0"]
            s, *_ = env.step(act((direction[0], direction[1], 0, 0)))
            new = objects(s, sp)["rover0"]
            if (old["x"], old["y"]) == (new["x"], new["y"]):
                print("REJECT", i, old); break
        print("END", objects(s, sp)["rover0"])
    env.close()


def operator_path(seed=0):
    env = make_env(); s, _ = env.reset(seed=seed); sp = env.observation_space
    # Move rover0 along explicit waypoints, reporting operator outcomes. Rover1 noops.
    def goto(x, y):
        nonlocal s
        for _ in range(80):
            q = objects(s, sp)["rover0"]
            dx = max(-.2, min(.2, x-q["x"])); dy = max(-.2, min(.2, y-q["y"]))
            if abs(dx)+abs(dy) < 1e-5: return
            old=(q["x"],q["y"]); s,*_=env.step(act((dx,dy,0,0)))
            nq=objects(s,sp)["rover0"]
            if old==(nq["x"],nq["y"]):
                # Try axis-separated detours.
                s,*_=env.step(act((dx,0,0,0)))
                if old==(objects(s,sp)["rover0"]["x"],objects(s,sp)["rover0"]["y"]):
                    s,*_=env.step(act((0,dy,0,0)))
        print("GOTOFAIL", x,y,objects(s,sp)["rover0"])
    d=objects(s,sp)
    for target in [d["sample0"], d["sample3"], d["objective0"], d["lander"]]:
        goto(target["x"], target["y"])
        print("AT", target, objects(s,sp)["rover0"])
        for sel,name in [(-5/6,"sample"),(-.5,"cal"),(-1/6,"image"),(.5,"send"),(.833333,"drop")]:
            before=objects(s,sp); s,r,t,tr,inf=env.step(act((0,0,0,sel))); after=objects(s,sp)
            changed=[n for n in after if after[n]!=before[n]]
            print(name,"changed",changed,"rover",after["rover0"])
    env.close()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv)>1 else "main"
    seed = int(sys.argv[2]) if len(sys.argv)>2 else 0
    {"main": main, "collision": collision_probe, "operators": operator_path}[mode](seed)
