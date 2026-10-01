from probe_vis_lib import *
import numpy as np
best=[]
for seed in range(24):
    env,obs,info=new_env(seed); L=layout(obs); env.close()
    for on in [n for n in L if n.startswith('objective')]:
        O=np.array([L[on]['x'],L[on]['y']])
        if O[0]<0.35: continue
        for pn in [f'obstacle{k}' for k in range(4,12)]:
            P=np.array([L[pn]['x'],L[pn]['y']])
            d=np.linalg.norm(P-O)
            if 0.3<d<1.1 and P[1]<O[1]-0.2:
                u=(P-O)/d
                r=O+1.9*u   # rover position on far side
                if r[0]>0.30 and abs(r[0])<2.25 and abs(r[1])<2.25:
                    best.append((d,seed,on,pn,tuple(np.round(O,3)),tuple(np.round(P,3)),tuple(np.round(r,3))))
best.sort()
for b in best[:6]: print(b)
np.save('q_best.npy',np.array([[b[0],b[1]] for b in best[:6]]))
