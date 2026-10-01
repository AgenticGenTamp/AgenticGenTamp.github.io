import numpy as np
exec(open('probe_kin_4.py').read().split('obs,_=env.reset')[0])
def objs(o): return {n:np.array([o.get(o.get_object_from_name(n),'x'),o.get(o.get_object_from_name(n),'y')]) for n in o.get_object_names() if n.startswith('small')}
for th,label in [(np.pi/2,'arm up'),(-np.pi/2,'arm down')]:
    obs,_=env.reset(seed=0)
    go(2.09,2.5,th); go(0.9,2.5); print(label,np.round(R(obs),3))
    o0=objs(obs)
    ys=[]
    for i in range(90):
        p=R(obs); st([0,-0.03,0,0,0]); ys.append(round(R(obs)[1],3))
        if np.allclose(R(obs),p): break
    o1=objs(obs)
    d={n:np.round(o1[n]-o0[n],3) for n in o0 if np.linalg.norm(o1[n]-o0[n])>1e-3}
    print(' final',np.round(R(obs),3),'steps',i,'moved',len(d),list(d.items())[:6])
    print(' fine',push([0,-0.001,0,0,0]))
