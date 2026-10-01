"""Search coarse Kinova shoulder/elbow poses for any floor-rod contact."""
import argparse
import itertools
import numpy as np
from env_client import make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env = make_env()
    try:
        state, _ = env.reset(seed=args.seed, options={"object_count": 1})
        rod = state.get_object_from_name("cuboid_0")
        robot = state.get_object_from_name("robot")
        def f(o, n): return float(state.get(o, n))
        def rp(): return np.array([f(rod,k) for k in ("x","y","z")])
        original = rp()

        # Put the shoulder roughly half a metre behind the rod.  Grid joint 1
        # as well because the exact arm mounting convention is unknown.
        goal = np.array([original[0]-.50, original[1]])
        for _ in range(12):
            err = goal-np.array([f(robot,"pos_base_x"),f(robot,"pos_base_y")])
            a=np.zeros(11,np.float32); a[:2]=np.clip(err/.87,-.1,.1); a[10]=1
            state,*_=env.step(a)

        home_tail = [np.pi, 0., -0.87, np.pi/2]
        poses = []
        for j1,j2,j4 in itertools.product((-1.0,0.,1.0), (-1.8,-.9,0.,.9,1.8), (-2.7,-1.8,-.9,0.,.9)):
            poses.append(np.array([j1,j2,home_tail[0],j4,home_tail[1],home_tail[2],home_tail[3]]))
        for pi,target in enumerate(poses):
            for k in range(9):
                q=np.array([f(robot,"pos_arm_joint%d"%j) for j in range(1,8)])
                err=(target-q+np.pi)%(2*np.pi)-np.pi
                a=np.zeros(11,np.float32); a[3:10]=np.clip(err*.7,-.1,.1); a[10]=1
                state,reward,term,trunc,_=env.step(a)
                move=np.linalg.norm(rp()-original)
                if move>.002:
                    print("HIT pose",pi,"substep",k,"target",np.round(target,3),
                          "q",np.round(q,3),"base",goal,"rod",np.round(rp(),4),"move",move,flush=True)
                    return
            if pi%10==0: print("pose",pi,"q",np.round(q,2),flush=True)
        print("NO_HIT",len(poses),"poses rod",rp(),flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
