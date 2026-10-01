import sys, numpy as np
from env_client import make_env
import diag_approach_fix as approach
from diag_approach_fix import GeneratedApproach, cube_face_yaw
np.set_printoptions(precision=3, suppress=True)
seed=int(sys.argv[1])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
def cs():
    return {n:(o['p'].round(3), round(float(np.degrees(cube_face_yaw(o['quat'])[1])),1), ap._which_bin(o['p'])) for n,o in ap.cubes.items()}
print('colors', {n:ap._cube_color(n) for n in ap.cube_names}, 'bins', {c:b.round(3) for c,b in ap.bins.items()})
print('init', cs())
bin0={c:b.copy() for c,b in ap.bins.items()}; prev=None; lastpos={n:o['p'].copy() for n,o in ap.cubes.items()}
for t in range(1000):
    a = ap.get_action(obs); ph=ap.phase
    if ph=='close' and ap.counter==1:
        c=ap.cubes[ap.target]; bw=ap._bracelet_world()
        u=np.array([np.cos(ap.phi),np.sin(ap.phi)]); v=np.array([-u[1],u[0]])
        d=bw[:2]-c['p'][:2]
        nb=[(n,round(float((o['p'][:2]-c['p'][:2])@u),3),round(float((o['p'][:2]-c['p'][:2])@v),3),round(float(o['p'][2]-c['p'][2]),3)) for n,o in ap.cubes.items() if n!=ap.target and np.linalg.norm(o['p'][:2]-c['p'][:2])<0.07]
        print(t,'CLOSE',ap.target,'along%.4f perp%.4f dz%.4f'%(d@u,d@v,bw[2]-c['p'][2]),'phi%.2f'%ap.phi,'clr%.3f'%ap._axis_clearance(ap.target,ap.phi),'nb',nb,'base',ap.base.round(3),'gam%.2f'%ap._gamma())
    obs, r, term, trunc, info = env.step(a)
    ap._parse(obs)
    if ap.phase!=prev:
        print(t, prev,'->',ap.phase, ap.target, 'tc', ap.cubes[ap.target]['p'].round(3) if ap.target else None, 'base', ap.base.round(3))
    prev=ap.phase
    for c,b in ap.bins.items():
        if np.linalg.norm(b-bin0[c])>0.01: print(t,'   BINMOVED',c,bin0[c].round(3),'->',b.round(3),'phase',ap.phase,'base',ap.base.round(3)); bin0[c]=b.copy()
    for n,o in ap.cubes.items():
        if n!=ap.target and np.linalg.norm(o['p']-lastpos[n])>0.01:
            print(t,'   MOVED',n,lastpos[n].round(3),'->',o['p'].round(3),'phase',ap.phase)
            lastpos[n]=o['p'].copy()
        elif n==ap.target: lastpos[n]=o['p'].copy()
    if term or trunc: break
print('final t',t,cs(), 'base',ap.base.round(3), 'bins', {c:b.round(3) for c,b in ap.bins.items()})
