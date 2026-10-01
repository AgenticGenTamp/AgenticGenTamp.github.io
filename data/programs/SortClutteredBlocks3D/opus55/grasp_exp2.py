import sys, numpy as np
from env_client import make_env
import approach
from approach import GeneratedApproach
np.set_printoptions(precision=3, suppress=True)
seeds=range(int(sys.argv[1]), int(sys.argv[2]))
env = make_env()
for seed in seeds:
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    prev=None; att=0
    for t in range(400):
        a = ap.get_action(obs)
        ph = ap.phase
        if ph=='close' and ap.counter==1:
            c=ap.cubes[ap.target]; bw=ap._bracelet_world()
            u=np.array([np.cos(ap.phi),np.sin(ap.phi)]); v=np.array([-u[1],u[0]])
            d=bw[:2]-c['p'][:2]
            nb=[(n, (o['p'][:2]-c['p'][:2])@u, (o['p'][:2]-c['p'][:2])@v, o['p'][2]-c['p'][2]) for n,o in ap.cubes.items() if n!=ap.target and np.linalg.norm(o['p'][:2]-c['p'][:2])<0.06]
            info0=dict(t=t, tgt=ap.target, along=round(d@u,4), perp=round(d@v,4), dz=round(bw[2]-c['p'][2],4), yaw=approach.cube_face_yaw(c['quat']), phi=round(ap.phi,3), nb=[(n,round(x,3),round(y,3),round(z,3)) for n,x,y,z in nb], c0=c['p'].copy())
        obs, r, term, trunc, info = env.step(a)
        if prev=='lift' and ap.phase!='lift':
            ok = ap.phase=='transport'
            print(seed, 'OK' if ok else 'FAIL', info0, 'cube now', ap.cubes[info0['tgt']]['p']-info0['c0'], flush=True)
            att+=1
            if att>=3: break
        prev=ap.phase
        if term: break
