import numpy as np, json
from env_client import make_env
from pngtool import readpng
env=make_env(); obs,_=env.reset(seed=0); obs=np.asarray(obs,float)
base=obs.copy()
# hide cubes 1..4 far away
for c in range(1,5):
    base[c*16+0]=20.0; base[c*16+1]=20.0; base[c*16+2]=20.0
base[147]=20; base[148]=20; base[149]=20   # wiper away
pts=[]
for x in [0.30,0.55,0.80,1.05]:
    for y in [-0.9,-0.3,0.3,0.9]:
        for z in [0.15,0.45,0.75]:
            pts.append((x,y,z))
res=[]
for p in pts:
    o=base.copy(); o[0],o[1],o[2]=p
    path=env.render_state(state=o.tolist(), label="cal")
    im=readpng(path).astype(int)
    r,g,b=im[:,:,0],im[:,:,1],im[:,:,2]
    m=(r>110)&(r-g>45)&(r-b>45)
    if m.sum()<3:
        res.append((p,None,int(m.sum()))); continue
    ys,xs=np.nonzero(m)
    res.append((p,(float(xs.mean()),float(ys.mean())),int(m.sum())))
env.close()
json.dump([[list(p),uv,n] for p,uv,n in res], open('cal_pts.json','w'))
ok=[r for r in res if r[1]]
print('found',len(ok),'/',len(res))
for p,uv,n in res[:6]: print(p,uv,n)
