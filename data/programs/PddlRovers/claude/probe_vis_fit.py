import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=39, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
obst=[(n,np.array([L[n]['x'],L[n]['y']]),L[n]['half_x'],L[n]['half_z']) for n in L if n.startswith('obstacle')]
env.close()
recs=np.load('vis_grid_39.npy')
def analyze(P):
    d=np.linalg.norm(O-P); u=(O-P)/d
    out=[]
    for n,q,hx,hz in obst:
        w=q-P; t=np.dot(w,u)
        if t<-0.1 or t>d+0.1: continue
        perp=abs(w[0]*u[1]-w[1]*u[0])
        if perp < hx+0.35:
            out.append((n,round(perp,3),round(t,3),round(hz,2),round(hx,2)))
    return d,out
print("INVISIBLE cells:")
for x,y,v in recs:
    if v: continue
    d,o=analyze(np.array([x,y])); print("  P=(%.2f,%.2f) d=%.3f  near-ray: %s"%(x,y,d,o))
print("VISIBLE cells with an obstacle within 0.15 of ray:")
for x,y,v in recs:
    if not v: continue
    d,o=analyze(np.array([x,y]))
    o=[z for z in o if z[1]<0.15]
    if o: print("  P=(%.2f,%.2f) d=%.3f  %s"%(x,y,d,o))
print("max visible dist", max(np.linalg.norm(O-np.array([x,y])) for x,y,v in recs if v))
print("min invisible dist", min(np.linalg.norm(O-np.array([x,y])) for x,y,v in recs if not v))
