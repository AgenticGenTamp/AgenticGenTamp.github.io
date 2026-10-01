import sys, numpy as np
from env_client import make_env
import diag_approach_head as A
np.set_printoptions(precision=3, suppress=True)
seed=int(sys.argv[1]); T=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(T):
    ph0=ap.phase
    if ph0=='select' and t>0:
        ap._parse(obs); cur=ap._bracelet_world(); gam=ap._gamma()
        print('t',t,'gam%.2f'%gam)
        for n in ap.cube_names:
            if ap._in_bin(n): continue
            c=ap.cubes[n]; yaw,tilt=A.cube_face_yaw(c['quat']); dmax=ap._delta_max(c['p'][:2])
            for k in range(2):
                phi=yaw+k*np.pi/2; clr=ap._axis_clearance(n,phi); need=A.wrap_half(phi-np.pi/2-gam-np.pi)
                rc=max(0,abs(need)-dmax)
                v=((3.0-20*clr if clr<0 else 0.0)-c['p'][2]*5+0.2*np.linalg.norm(c['p'][:2]-cur[:2])+0.4*rc+0.5*ap.fail.get(n,0)+0.2*tilt)
                print('  ',n,'phi%.2f clr%.3f need%.2f dmax%.2f rc%.2f fail%d v%.3f'%(phi,clr,need,dmax,rc,ap.fail.get(n,0),v))
    a=ap.get_action(obs); obs,r,term,trunc,info=env.step(a)
