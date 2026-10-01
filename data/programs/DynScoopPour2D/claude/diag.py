from env_client import make_env
import numpy as np, sys
import approach as A
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
env=make_env(); obs,info=env.reset(seed=seed)
ap=A.GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
names=[n for n in obs.get_object_names() if n.startswith('small')]
def inpan():
    R=obs.get_object_from_name('robot'); x=float(obs.get(R,'x')); y=float(obs.get(R,'y'))
    cy=y-ap.rel_f
    return sum(1 for n in names if x-0.56<float(obs.get(obs.get_object_from_name(n),'x'))<x+0.06 and cy-0.03<float(obs.get(obs.get_object_from_name(n),'y'))<cy+0.55)
ph=None
for t in range(1000):
    a=ap.get_action(obs); obs,r,te,tr,info=env.step(a)
    if ap.phase!=ph:
        nr=sum(1 for n in names if float(obs.get(obs.get_object_from_name(n),'x'))>1.8)
        print("t=%d %s inpan=%d right=%d relf=%.2f pan_y=%.2f"%(t,ap.phase,inpan(),nr,ap.rel_f,ap._pan_y()),flush=True)
        ph=ap.phase
    if te: break
print("end",t,te)
