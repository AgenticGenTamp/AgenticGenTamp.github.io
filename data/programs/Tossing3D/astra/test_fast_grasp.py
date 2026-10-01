from env_client import make_env
from grasp_policy import GeneratedApproach
import numpy as np
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hover-z', type=float, default=.12)
parser.add_argument('--hover', type=int, default=30)
parser.add_argument('--descend', type=int, default=20)
parser.add_argument('--close', type=int, default=8)
parser.add_argument('--yaw-threshold', type=float, default=-.2)
parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
parser.add_argument('--counts', type=int, nargs='+', default=[1, 2])
parser.add_argument('--detail', action='store_true')
parser.add_argument('--piecewise', action='store_true')
args = parser.parse_args()

class FastGrasp(GeneratedApproach):
    def reset(self, state, info):
        super().reset(state, info)
        self.cube = self.cubes[0]
        self.yaw = 2*np.arctan2(state.get(self.cube, 'qz'), state.get(self.cube, 'qw'))
        if args.piecewise:
            self.yaw = self.yaw if self.yaw < -.2 or self.yaw > .6 else 0.
        else:
            self.yaw = self.yaw if self.yaw < args.yaw_threshold else 0.
        self.q = self.ik(.55, args.hover_z)
    def get_action(self, state):
        self.t += 1
        a = np.zeros(18)
        base = np.array([state.get(self.robot, f) for f in ('pos_base_x', 'pos_base_y', 'pos_base_rot')])
        p = np.array([state.get(self.cube, f) for f in ('x', 'y', 'z')])
        if self.t <= args.hover:
            a[:3] = np.clip(np.array([p[0]-.65, p[1]-.00135, 0])-base, -.1, .1)
        if self.t == args.hover+1:
            self.q = self.ik(.55, .025)
        if self.t >= args.hover+args.descend+1:
            a[10] = 1
        if self.t == args.hover+args.descend+args.close+1:
            self.q = self.ik(.55, .65)
        q = np.array([state.get(self.robot, 'pos_arm_joint'+str(j)) for j in range(1, 8)])
        err = self.q-q
        a[3:10] = np.clip(err, -.1, .1)
        a[11:18] = 5*err
        return a.astype(np.float32)

for count in args.counts:
    for seed in args.seeds:
        env = make_env()
        state, info = env.reset(seed=seed, options={'object_count': count})
        policy = FastGrasp(env.action_space, env.observation_space, {})
        policy.reset(state, info)
        initial = [state.get(policy.cube, f) for f in ('x', 'y', 'z')]
        start = time.time()
        for step in range(args.hover+args.descend+args.close+18):
            state, reward, terminated, truncated, info = env.step(policy.get_action(state))
            if args.detail and step % 5 == 4:
                print('TRACE', seed, step+1, [round(state.get(policy.cube,f),4) for f in ('x','y','z')], flush=True)
            if terminated or truncated:
                break
        final = [state.get(policy.cube, f) for f in ('x', 'y', 'z')]
        print(json.dumps({'seed': seed, 'count': len(policy.cubes), 'initial': initial,
                          'final': final, 'grasp': final[2]>.4, 'seconds': time.time()-start}), flush=True)
        env.close()
