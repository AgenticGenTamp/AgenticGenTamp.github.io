import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=39, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
obst=[(n,np.array([L[n]['x'],L[n]['y']]),L[n]['half_x'],L[n]['half_y'],L[n]['z'],L[n]['half_z']) for n in L if n.startswith('obstacle')]
env.close()
recs=np.load('vis_grid_39.npy')
def pred(P,H=0.5,pad=0.0,R=2.0,detail=False):
    d=np.linalg.norm(O-P); u=(O-P)/d
    if d>R: return False,'range'
    for n,q,hx,hy,z,hz in obst:
        w=q-P; t=np.dot(w,u); perp=abs(w[0]*u[1]-w[1]*u[0])
        if not (0<t<d): continue
        if perp>hx+pad: continue
        hray=H+(0.201-H)*(t/d)
        if hray< z+hz: return False,'%s perp=%.3f frac=%.2f hray=%.3f'%(n,perp,t/d,hray)
    return True,''
print("ERRORS of H=0.5,pad=0,R=2.0 model:")
for x,y,v in recs:
    p,why=pred(np.array([x,y]))
    if p!=bool(v):
        print("  (%.1f,%.1f) obs_vis=%d pred=%d  %s  d=%.3f"%(x,y,v,p,why,np.linalg.norm(O-np.array([x,y]))))
