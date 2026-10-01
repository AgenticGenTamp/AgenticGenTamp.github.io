"""Seed-0 red placement followed by a direct mirrored blue rake."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np

EDGE = np.array([0., 1.3, np.pi, -1.7, 0., 1., 0.])
HOME = np.array([0., -.349, np.pi, -2.548, 0., -.873, np.pi / 2])


def vals(s, name, fs):
    o = s.get_object_from_name(name)
    return np.array([float(s.get(o, f)) for f in fs])


def robot(s):
    return vals(s, "robot", ["pos_base_x", "pos_base_y", "pos_base_rot"] +
                ["pos_arm_joint%d" % i for i in range(1, 8)])


def xy(s, name):
    return vals(s, name, ["x", "y"])


def action(s, base, q, limit=.1):
    r = robot(s); a = np.zeros(11, np.float32)
    err = np.asarray(base) - r[:3]
    err[2] = (err[2] + np.pi) % (2*np.pi) - np.pi
    a[:3] = np.clip(1.2*err, -limit, limit)
    a[3:10] = np.clip(.7*(q-r[3:10]), -.1, .1)
    return a


def run(env, s, n, base, q, limit=.1):
    out = None
    for _ in range(n):
        s, *out = env.step(action(s, base, q, limit))
    return s, out


def main():
    env = make_env(); s, info = env.reset(seed=0, options={"object_count": 4})
    targets = {c: xy(s, "bin_"+c).copy() for c in ("red","green","blue","yellow")}
    # Use the already validated first 275 policy steps to place red.
    p = GeneratedApproach(env.action_space, env.observation_space, {}); p.reset(s, info)
    for _ in range(275): s, reward, term, trunc, info = env.step(p.get_action(s))
    print("RED", np.round(xy(s,"cube1"),4), np.linalg.norm(xy(s,"cube1")-targets["red"]))
    # Fold, route outside the lower table edge, then mirror red about origin.
    b = robot(s)[:3].copy(); s,_=run(env,s,45,b,HOME)
    s,_=run(env,s,45,[0.,-.85,b[2]],HOME)
    s,_=run(env,s,45,[1.05,-.85,np.pi],HOME)
    c0=xy(s,"cube3").copy(); y=float(c0[1]-.042)
    s,_=run(env,s,45,[1.05,y,np.pi],HOME)
    s,_=run(env,s,130,[1.05,y,np.pi],EDGE)
    # Continue farther than the generic controller, latching only cube3.
    hit=None
    for k in range(80):
        base=[.55,y,np.pi]
        a=action(s,base,EDGE,.012)
        s,reward,term,trunc,info=env.step(a)
        if np.linalg.norm(xy(s,"cube3")-c0)>.001:
            hit=robot(s)[:3].copy(); print("HIT",k,np.round(hit,4),np.round(xy(s,"cube3"),4)); break
    if hit is not None:
        qt=EDGE.copy(); qt[0]=.9
        for k in range(100):
            s,reward,term,trunc,info=env.step(action(s,hit,qt))
            if k%10==0: print("SWEEP",k,np.round(xy(s,"cube3"),4),round(float(reward),3))
    print("FINAL", "red",round(float(np.linalg.norm(xy(s,"cube1")-targets["red"])),5),
          "blue",round(float(np.linalg.norm(xy(s,"cube3")-targets["blue"])),5),
          "cube3",np.round(xy(s,"cube3"),5),"reward",reward,"term",term)
    env.close()


if __name__ == "__main__": main()
