"""Reduced search for seed 101 green grasp after rolled clearance posture."""
import itertools
import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def g(s, name, feature):
    return float(s.get(s.get_object_from_name(name), feature))


def move_joint(env, s, index, goal, p, grip=1.0, step=.2):
    feature = "joint_" + str(index - 2)
    for _ in range(30):
        cur = g(s, "robot", feature)
        d = p.w(goal-cur) if index in (7, 9) else goal-cur
        if abs(d) < .002:
            break
        a = np.zeros(11, np.float32); a[index] = np.clip(d, -step, step); a[10] = grip
        s, *_ = env.step(a)
    return s


def setup():
    env = make_env(); s, info = env.reset(seed=101)
    p = GeneratedApproach(env.action_space, env.observation_space, {}); p.reset(s, info)
    for _ in range(150):
        if p.stage == 5:
            break
        s, *_ = env.step(p.get_action(s))
    # stage 5's lifted base positioning, stopping before arm descent.
    for _ in range(40):
        old = np.r_[p.robot(s), g(s,"robot","base_rot")]
        a = p.get_action(s)
        # Do not execute an action which starts increasing q2 (the descent).
        if a[4] > .01:
            break
        s, *_ = env.step(a)
        if np.max(abs(np.r_[p.robot(s),g(s,"robot","base_rot")]-old)) < 1e-7:
            break
    # Requested clearance: nominal q2, joint5 +1.4 and joint7 +.4.
    s = move_joint(env, s, 7, p.w(p.Q[4]+1.4), p)
    s = move_joint(env, s, 9, p.w(p.Q[6]+.4), p)
    s = move_joint(env, s, 4, p.Q[1], p, step=.05)
    return env, s, p


def trial(yaw_delta, dx, dy, j7_delta, verbose=False):
    env, s, p = setup()
    start = [g(s,"robot",f) for f in ("base_x","base_y","base_rot")]
    target = np.array([start[0]+dx, start[1]+dy])
    yaw = p.w(start[2]+yaw_delta)
    j7 = p.w(p.Q[6]+.4+j7_delta)
    for _ in range(25):
        a=np.zeros(11,np.float32)
        a[0]=np.clip(target[0]-g(s,"robot","base_x"),-.05,.05)
        a[1]=np.clip(target[1]-g(s,"robot","base_y"),-.05,.05)
        a[2]=np.clip(p.w(yaw-g(s,"robot","base_rot")),-.05,.05)
        a[9]=np.clip(p.w(j7-g(s,"robot","joint_7")),-.05,.05)
        a[10]=1.; old=np.array([g(s,"robot",f) for f in ("base_x","base_y","base_rot")])
        s,*_=env.step(a)
        if max(abs(a[[0,1,2,9]]))<.002: break
    a=np.zeros(11,np.float32);a[10]=-1.;s,*_=env.step(a)
    hit=g(s,"green0","grasp_active")>.5
    if hit or verbose:
        vals=[g(s,"robot",f) for f in ("base_x","base_y","base_rot","joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7","grasp_active")]
        print("HIT" if hit else "MISS", "delta",yaw_delta,dx,dy,j7_delta,"state",[round(x,7) for x in vals], flush=True)
    env.close(); return hit


if __name__ == "__main__":
    # Coarse coupled search. Translation compensation is expressed in world axes.
    yaws=(-.16,-.08,0.,.08,.16)
    xys=tuple(itertools.product((-.08,-.04,0.,.04,.08),(-.08,-.04,0.,.04,.08)))
    j7s=(-.4,-.2,0.,.2,.4)
    count=0
    for yd,(dx,dy),j7d in itertools.product(yaws,xys,j7s):
        count+=1
        if trial(yd,dx,dy,j7d):
            print("FOUND after",count);break
    else: print("NONE",count)
