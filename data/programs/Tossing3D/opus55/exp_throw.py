import sys, json
import numpy as np
from env_client import make_env
import toss as T

args = dict(seed=0, q7=0.0, w=3.0, k=1, qs=[-0.5, 0.8, 0.8], bx=0.3, nsw=6, sc=[1, -1, -1], bvx=0.0)
for s in sys.argv[1:]:
    k, v = s.split('=')
    args[k] = json.loads(v)
env = make_env()
obs, _ = env.reset(seed=args['seed'])
name = 'cube_0'


def run():
    global obs
    g = T.pick(obs, name)
    a = next(g)
    while True:
        obs, r, te, tr, _ = env.step(a)
        try:
            a = g.send(obs)
        except StopIteration:
            break
    print('picked', T.objpos(obs, name).round(3))
    qs = args['qs']
    Qs = T.plane_q(obs, qs[0], qs[1], qs[2], args['q7'])
    # move to windup while driving base
    goal = np.array([args['bx'], 0.0, 0.0])
    if args.get('aim'):
        bp = T.objpos(obs, 'bin_0')
        goal = np.array([bp[0] - args['aim'], bp[1] - args.get('yoff', 0.0), 0.0])
        print('bin', bp.round(3), 'goal', goal.round(3))
    for i in range(150):
        a = T.arm_act(obs, Qs, T.CLOSE, vmax=0.08, base=T.base_delta(obs, goal))
        obs, r, te, tr, _ = env.step(a)
        if np.abs(T.rq(obs) - Qs).max() < 0.01 and np.linalg.norm(T.rb(obs) - goal) < 0.01:
            break
    for _ in range(3):
        obs, *_ = env.step(T.arm_act(obs, Qs, T.CLOSE))
    pf, Rf = T.kin.fk(*T.rb(obs), T.rq(obs)); off = Rf.T @ (T.objpos(obs, name) - pf)
    print('OFF', args['seed'], off.round(4), 'yaw', round(T.wrap(T.obj_yaw(obs, name)), 3))
    print('windup', i, 'cube', T.objpos(obs, name).round(3), 'base', T.rb(obs).round(3))
    w = args['w'] * 0.1
    dq = np.zeros(7); dq[[1, 3, 5]] = np.array(args['sc']) * w
    hold = T.rq(obs).copy()
    traj = []; rews = []; terms = []
    for t in range(args['nsw'] + 40):
        grip = T.OPEN if t >= args['k'] else T.CLOSE
        if t < args['nsw']:
            a = T.vel_act(obs, dq, hold, grip)
        else:
            a = T.vel_act(obs, np.zeros(7), hold, grip)
            if t == args['nsw']:
                hold = T.rq(obs).copy()
            a = T.arm_act(obs, hold, grip)
        obs, r, te, tr, _ = env.step(a)
        p = T.objpos(obs, name); v = T.objvel(obs, name)
        traj.append(np.r_[p, v])
        rews.append(r); terms.append(te)
        if te: break
        if args.get('v'):
            print(t, 'q', T.rq(obs)[[1, 3, 5]].round(2), 'cube', p.round(3), 'v', v.round(2), 'r', round(r, 3), te, 'bin', T.objpos(obs, 'bin_0').round(3), T.objvel(obs, 'bin_0').round(2))
    traj = np.array(traj)
    print('rews', np.round(rews, 3).tolist(), 'term', any(terms), 'bin_after', T.objpos(obs, 'bin_0').round(3))
    land = np.argmax(traj[:, 2] < 0.06) if (traj[:, 2] < 0.06).any() else -1
    b = T.rb(obs)
    print('RES', json.dumps({k: args[k] for k in ('w', 'k', 'qs', 'q7', 'bx', 'bvx')}), 'rel_v', traj[args['k'] + 1, 3:].round(2),
          'land', traj[land, :3].round(3), 'final', traj[-1, :3].round(3), 'D', round(traj[land, 0] - b[0], 3))


run()
env.close()
