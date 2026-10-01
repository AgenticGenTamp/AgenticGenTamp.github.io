import numpy as np
from env_client import make_env
import approach as A
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':4})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
def placed(obs,name):
    c=A.obj_pos(obs,name); bp=ap._bin_for(name)
    return abs(c[0]-bp[0])<0.045 and abs(c[1]-bp[1])<0.045 and c[2]>0.405
T=0; hist=[]
def snap():
    b,q,g=A.robot_state(obs)
    d={n:np.round(A.obj_pos(obs,n),3).tolist() for n in sorted(obs.get_object_names()) if n.startswith(('cube','bin'))}
    return (T, ap.phase, np.round(b,3).tolist(), d)
for rnd in range(3):
    t=0
    while ap.idx < len(ap.order) and t<1200 and T<1500:
        a=ap.get_action(obs); obs,r,te,tr,i=env.step(a); t+=1; T+=1
        hist.append(snap())
        if te or tr:
            print('TERM at',T,'te',te,'tr',tr,'r',r,flush=True)
            for h in hist[-6:]: print(h,flush=True)
            raise SystemExit
    bad=[n for n in ap.order if not placed(obs,n)]
    print('round',rnd,'T',T,'bad',bad,flush=True)
    if not bad: break
    ap.order=bad; ap.idx=0; ap.phase='app'; ap.t=0; ap.bias=np.zeros(3); ap.target=None
print('no term', T)
