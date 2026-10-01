import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=1, options={'object_count':1})
L=layout(obs); lan=np.array([L['lander']['x'],L['lander']['y']])
for P,lbl in [((0.35,-2.10),'SEND OK'),((0.23,-1.75),'SEND FAIL'),((0.71,-1.35),'FAIL far')]:
    P=np.array(P); d=np.linalg.norm(lan-P); u=(lan-P)/d
    near=[]
    for n in L:
        if not n.startswith('obstacle'): continue
        q=np.array([L[n]['x'],L[n]['y']]); w=q-P; t=np.dot(w,u); pe=abs(w[0]*u[1]-w[1]*u[0])
        if 0<t<d and pe<L[n]['half_x']+0.25:
            near.append((n,round(pe,3),round(t/d,2),round(L[n]['half_z'],2)))
    print("%-10s P=%s d=%.2f near-ray: %s"%(lbl,P,d,near))
env.close()
