import numpy as np, sys
from env_client import make_env
from diag_approach_fix import GeneratedApproach
np.set_printoptions(precision=3, suppress=True)
env = make_env()
seed=int(sys.argv[1]); T=int(sys.argv[2]); every=int(sys.argv[3]) if len(sys.argv)>3 else 2
obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(T):
    a = ap.get_action(obs)
    ph=ap.phase
    obs, r, term, trunc, info = env.step(a)
    if t % every == 0 or ph in ('close','lift','release'):
        bw = ap._bracelet_world()
        c = ap.cubes.get(ap.target, {}).get('p') if ap.target else None
        print(t, ph, ap.target, 'xyerr%.4f'%np.linalg.norm(ap._bracelet_world()[:2]-ap.cubes[ap.target]['p'][:2]), 'bin', ap.bins['red'].round(3), 'phi%.2f'%ap.phi, 'bw', bw, 'c', c, 'base', ap.base, 'gam %.2f'%ap._gamma(), 'a', a[[0,1,2,9,10]])
