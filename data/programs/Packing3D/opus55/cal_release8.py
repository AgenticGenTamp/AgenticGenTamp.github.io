from cal_trial import *
import sys
def test(tx,ty,pz):
    t=trial(0,'part0',0,0,0.05); d=t['d']; b=d.r['base']; q=d.r['q']
    path=[[0.313,0.289,0.3],[tx,ty,0.3]]+[[tx,ty,z] for z in np.arange(0.28,pz+0.05,-0.02)]+[[tx,ty,pz+0.05]]
    for tgt in path:
        qs,_,_=ik(tgt,q,base=b); ok=d.goto_q(qs,grip=0.0); q=d.r['q']
        if not ok: break
    p=part(d.obs,'part0')[0]
    d.step(act(grip=1.0)); p2=part(d.obs,'part0')[0]
    for i in range(2): d.step(act())
    p3=part(d.obs,'part0')[0]
    print(f'xy=({tx},{ty}) pz_cmd={pz:.3f} reached={ok} part_z={p[2]:.4f} ga_after_open={d.r["ga"]} part_after={np.round(p3[:3],4)}',flush=True)
for tx,ty,pzs in [(0.3,0.0,[0.105,0.095,0.085,0.07]),(0.1,0.2,[0.093,0.087,0.08,0.06]),(0.3,0.25,[0.087]),(-0.1,0.35,[0.087,0.2])]:
    for pz in pzs: test(tx,ty,pz)
