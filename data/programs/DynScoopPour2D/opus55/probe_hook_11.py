from probe_hook_lib import *
p=P()
def smalls():
    d={}
    for n in p.obs.get_object_names():
        if n.startswith('small'):
            o=p.obs.get_object_from_name(n); d[n]=(float(p.obs.get(o,'x')),float(p.obs.get(o,'y')))
    return d
p.goto(th=-np.pi/2); p.goto(y=1.3); p.goto(x=3.105); p.goto(y=0.74)
p.step([0,0,0,0,-0.015],11)
p.goto(y=1.0); s0=smalls()
p.goto(x=2.01); print('at wall',p.r(),p.h())
s1=smalls(); print('moved while sliding in air:',{k:(round(s1[k][0]-s0[k][0],3),round(s1[k][1]-s0[k][1],3)) for k in s0 if np.hypot(s1[k][0]-s0[k][0],s1[k][1]-s0[k][1])>1e-3})
for k in range(40):
    y0=p.r()['y']; p.step([0,-0.01,0,0,0])
    if p.r()['y']==y0: break
s2=smalls(); print('lowered',p.r(),p.h())
print('moved:',{k:(round(s2[k][0]-s1[k][0],3),round(s2[k][1]-s1[k][1],3)) for k in s0 if np.hypot(s2[k][0]-s1[k][0],s2[k][1]-s1[k][1])>1e-3})
