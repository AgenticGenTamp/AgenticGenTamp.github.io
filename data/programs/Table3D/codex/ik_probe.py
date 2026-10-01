"""Test Panda forward-kinematics hypotheses against live grasp activation."""

import argparse
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from env_client import make_env


ORIGINS = [
    ([0, 0, .333], [0, 0, 0]), ([0, 0, 0], [-np.pi/2, 0, 0]),
    ([0, -.316, 0], [np.pi/2, 0, 0]), ([.0825, 0, 0], [np.pi/2, 0, 0]),
    ([-.0825, .384, 0], [-np.pi/2, 0, 0]), ([0, 0, 0], [np.pi/2, 0, 0]),
    ([.088, 0, 0], [np.pi/2, 0, 0]),
]
LO = np.array([-2.85, -1.75, -3.2, -3.0, -2.85, -0.05, -2.85])
HI = np.array([ 2.85,  1.75,  3.2, -0.1,  2.85,  3.70,  2.85])
STATE_OFFSET = np.array([0., 0., np.pi, 0., 0., np.pi, -np.pi / 4])


def fk(q):
    t = np.eye(4)
    for qi, (xyz, rpy) in zip(q, ORIGINS):
        a = np.eye(4); a[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix(); a[:3, 3] = xyz
        z = np.eye(4); z[:3, :3] = Rotation.from_euler("z", qi).as_matrix()
        t = t @ a @ z
    return (t @ np.array([0, 0, .107, 1]))[:3]


def fk_pose(q):
    t = np.eye(4)
    for qi, (xyz, rpy) in zip(q, ORIGINS):
        a = np.eye(4); a[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix(); a[:3, 3] = xyz
        z = np.eye(4); z[:3, :3] = Rotation.from_euler("z", qi).as_matrix()
        t = t @ a @ z
    return (t @ np.array([0, 0, .107, 1]))[:3], t[:3, :3]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=0); args=ap.parse_args()
    env=make_env(); s,_=env.reset(seed=args.seed); rng=np.random.default_rng(7)
    cubes=[s.get_object_from_name(n) for n in sorted(s.get_object_names()) if n.startswith('cube')]
    robot=s.get_object_from_name('robot'); tested=0
    # Hypotheses map world cube positions to the FK base frame.
    nominal=np.array([0.,-.35,0.,-2.5,0.,2.2716,.7854])
    for zbase in [0., .4]:
      for sx in [1., -1.]:
       for sy in [1., -1.]:
        for cube in cubes:
         world=np.array([s.get(cube,'pose_x'),s.get(cube,'pose_y'),s.get(cube,'pose_z')+.015])
         goal=np.array([sx*world[0],sy*world[1],world[2]-zbase])
         for k in range(2):
          x0=np.clip(nominal+rng.normal(0,.25,7),LO,HI)
          def residual(q):
            p,r=fk_pose(q)
            return np.r_[p-goal, .35*r[:2,2]]
          sol=least_squares(residual,x0,bounds=(LO,HI),max_nfev=250)
          if np.linalg.norm(fk(sol.x)-goal)>.012 or np.linalg.norm(fk_pose(sol.x)[1][:2,2])>.15: continue
          target=sol.x-STATE_OFFSET; tested+=1
          for _ in range(16):
            robot=s.get_object_from_name('robot')
            cur=np.array([s.get(robot,f'joint_{i}') for i in range(1,8)])
            a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(target-cur,-.4,.4); a[10]=1
            s,_,term,trunc,_=env.step(a)
            if term or trunc: break
            if np.max(np.abs(target-cur))<.01: break
          a=np.zeros(11,dtype=np.float32); a[10]=-1
          s,_,term,trunc,_=env.step(a); robot=s.get_object_from_name('robot')
          if s.get(robot,'grasp_active')>.5:
            print('GRASP',zbase,sx,sy,cube.name,target.tolist(),fk(target).tolist()); env.close(); return
          if term or trunc: print('ENDED'); env.close(); return
    print('NO_GRASP tested',tested); env.close()


if __name__=='__main__': main()
