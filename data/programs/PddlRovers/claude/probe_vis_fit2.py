import numpy as np, itertools
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=39, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
obst=[(n,np.array([L[n]['x'],L[n]['y']]),L[n]['half_x'],L[n]['half_y'],L[n]['z'],L[n]['half_z']) for n in L if n.startswith('obstacle')]
env.close()
recs=np.load('vis_grid_39.npy')
print("INVISIBLE:")
for x,y,v in recs:
    if v: continue
    P=np.array([x,y]); d=np.linalg.norm(O-P); u=(O-P)/d
    info=[]
    for n,q,hx,hy,z,hz in obst:
        if hz<0.1: continue
        w=q-P; t=np.dot(w,u); perp=abs(w[0]*u[1]-w[1]*u[0])
        if 0<t<d and perp<0.25: info.append((n,round(perp,3),round(t,3),round(t/d,2)))
    print("  (%.1f,%.1f) d=%.3f pillars:%s"%(x,y,d,info))
# fit model: blocked iff exists pillar with perp<=pad and ray height at pillar < 0.401
best=None
for H in np.arange(0.2,1.21,0.02):
  for pad in np.arange(0.0,0.31,0.01):
    for R in [1.9,2.0,2.05,2.1,2.5,3.0,4.0,10.0]:
      err=0
      for x,y,v in recs:
        P=np.array([x,y]); d=np.linalg.norm(O-P); u=(O-P)/d
        pred = d<=R
        if pred:
          for n,q,hx,hy,z,hz in obst:
            w=q-P; t=np.dot(w,u); perp=abs(w[0]*u[1]-w[1]*u[0])
            if not (0<t<d): continue
            if perp> hx+pad: continue
            hray = H + (0.201-H)*(t/d)
            if hray < z+hz: pred=False; break
        err += (pred != bool(v))
      if best is None or err<best[0]: best=(err,H,pad,R)
print("best fit err=%d/%d  H=%.2f pad=%.2f R=%s"%(best[0],len(recs),best[1],best[2],best[3]))
