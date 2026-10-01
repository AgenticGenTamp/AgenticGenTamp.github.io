from exprot import *
maxerr=0; n=0
for seed in range(4):
    obs,acts=to_grasp(seed)
    r,s,_=rd(obs); print('seed',seed,'robot',{k:round(v,3) for k,v in r.items()},'stick',[round(v,3) for v in s])
    for dth in [0.1,-0.1,0.196,-0.196,0.05,-0.03]*3:
        r,s,_=rd(obs); o2=step(obs,dth=dth); r2,s2,_=rd(o2)
        rej = abs(wrap(r2['theta']-r['theta']))<1e-6
        if rej: print('  rejected dth',dth,'stick',[round(v,3) for v in s]); obs=o2; continue
        p=pred(r,s,dth); e=max(abs(p[0]-s2[0]),abs(p[1]-s2[1]),abs(wrap(p[2]-s2[2])))
        dth_act=wrap(r2['theta']-r['theta'])
        other=max(abs(r2['x']-r['x']),abs(r2['y']-r['y']),abs(r2['arm_joint']-r['arm_joint']))
        maxerr=max(maxerr,e); n+=1
        print('  dth %.3f actual %.4f err %.2e base/arm change %.1e vac %.0f'%(dth,dth_act,e,other,r2['vacuum']))
        obs=o2
print('MAX ERR',maxerr,'n',n)
