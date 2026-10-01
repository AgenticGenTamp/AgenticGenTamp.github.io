import math
import numpy as np
from env_client import make_env


def f(s, o, k): return float(s.get(o, k))

def run(seed, waypoints, arm=-0.08):
    env = make_env(); s, info = env.reset(seed=seed)
    rob = s.get_objects(env.observation_space.get_type("kin_robot"))[0]
    names = [n for n in s.get_object_names() if n.startswith("small")]
    steps = 0
    term = False
    for tx, ty, hold in waypoints:
        for _ in range(hold):
            x, y = f(s, rob, "x"), f(s, rob, "y")
            a = np.array([np.clip(tx-x,-.03,.03), np.clip(ty-y,-.03,.03), 0, arm, 0], dtype=np.float32)
            s, r, term, trunc, inf = env.step(a); steps += 1
            if term or trunc: break
        xs=[f(s,s.get_object_from_name(n),"x") for n in names]
        ys=[f(s,s.get_object_from_name(n),"y") for n in names]
        print("phase",(tx,ty),"robot",round(f(s,rob,"x"),2),round(f(s,rob,"y"),2),"x>1.75",sum(x>1.75 for x in xs),"xmax",round(max(xs),2),"ymax",round(max(ys),2),"term",term)
        if term or trunc: break
    env.close(); return steps, term


paths = {
 "low_then_up": [(1.,2.1,60),(.08,2.1,40),(.08,.25,80),(1.35,.25,70),(1.35,2.05,80),(2.3,2.05,50)],
 "up_columns": [(1.,2.1,60),(.1,2.1,40),(.1,.2,80),(.45,.2,30),(.45,2.0,70),(.8,.2,70),(.8,2.,70),(1.15,.2,70),(1.15,2.,70),(1.5,.2,70),(1.5,2.,70),(2.2,2.,40)],
 "base_sweep": [(1.,2.1,60),(.08,2.1,40),(.08,.55,80),(1.55,.55,70),(1.55,1.7,50),(2.3,1.7,40)],
}
for i,(name,path) in enumerate(paths.items()):
    print("PATH",name)
    print(run(100+i,path))
