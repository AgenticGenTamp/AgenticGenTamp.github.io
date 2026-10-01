from cal_trial import *
def s(lbl,t): print(f"{lbl} reached={t['reached']} tcp_rel={np.round(t['tcp'],3)} ga={t['ga']} gtf={np.round(t['gtf'],3)}",flush=True)
for seed,pn in [(0,'part1'),(1,'part0'),(5,'part1')]:
    d=Driver(seed); pp,_=part(d.obs,pn); print(seed,pn,np.round(pp[:3],3), d.obs.get(d.obs.get_object_from_name(pn),'triangle_type'))
    s(f'seed{seed} {pn} center',trial(seed,pn,0,0,0.05,d=d))
