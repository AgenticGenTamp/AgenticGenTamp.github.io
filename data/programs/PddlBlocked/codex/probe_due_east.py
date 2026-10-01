"""Probe due-east blocker grasps without modifying the submitted policy."""
import math
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def g(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def run(seed, theta, side, offset_y=0.0, west=True, switch=None):
    env = make_env(); s, info = env.reset(seed=seed)
    p = GeneratedApproach(env.action_space, env.observation_space, {})
    p.reset(s, info); p.theta = theta; p.west_branch = west
    c, z = math.cos(theta), math.sin(theta)
    p.off = np.array([c*p.OFF[0]-z*p.OFF[1], z*p.OFF[0]+c*p.OFF[1]])
    t = p.target(p.blocker)+np.array([0., offset_y])
    p.approach_route = [np.array([3.4, side]), np.array([t[0], side]), t]
    last = None
    for k in range(220):
        if p.stage == 5 and switch is not None and not hasattr(p, '_switched'):
            p._switched=True;p.theta=switch;p.west_branch=True
            c,z=math.cos(switch),math.sin(switch)
            p.off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]])
            final=p.target(p.green)+np.array([c*.10+z*.125,z*.10-c*.125])
            p.se_green_route=[np.array([3.4,1.6]),np.array([3.4,-1.6]),
                              np.array([final[0],-1.6]),final]
            p.stage=55
        old = p.robot(s).copy(); a = p.get_action(s)
        s, _, done, trunc, _ = env.step(a)
        if p.stage != last:
            last = p.stage
            print('stage', k, last, 'base', p.robot(s).round(4), 'held', g(s,'robot','grasp_active'))
        if done or trunc: break
    print('RESULT',seed,theta,side,offset_y,'done',done,'k',k,'stage',p.stage,
          'base',p.robot(s).round(4),'blockheld',g(s,'blocker','grasp_active'),
          'greenheld',g(s,'green0','grasp_active'))
    env.close()


if __name__ == '__main__':
    run(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
        float(sys.argv[4]) if len(sys.argv)>4 else 0.,
        not (len(sys.argv)>5 and sys.argv[5] == 'normal'),
        float(sys.argv[6]) if len(sys.argv)>6 else None)
