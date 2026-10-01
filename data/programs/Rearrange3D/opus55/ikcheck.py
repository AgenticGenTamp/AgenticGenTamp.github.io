from probe_lib import *
pr=P(0); obj=16
c=pr.obs[obj:obj+3].copy()
b=pr.obs[93:96]
for z in [0.66,0.55,0.53]:
    q,e=kin.ik_arm(pr.ta([c[0],c[1],z]),kin.rotz(-b[2])@Rdown(1.57),pr.QT,iters=300)
    print(z,'err',e,'q',q.round(3),'fk',pr.fk(q).round(3))
pr.goto(q); print('reached fk',pr.fk().round(3), 'q',pr.obs[96:103].round(3))
