import numpy as np
from env_client import make_env
from pngtool import readpng
from probe_drawer_crop import writepng
M=np.load('cam_M.npy')
def pr(P):
    h=M@np.r_[np.asarray(P,float),1.0]; return h[0]/h[2], h[1]/h[2]
env=make_env(); obs,_=env.reset(seed=0)
st=obs.copy(); st[125]=3.0; st[126]=-2.0; st[147:163]=0.0
img=readpng(env.render_state(state=st.tolist(),label="ov_base")).astype(int)
# validate on cubes
print('cube check (world -> proj px, colour at px):')
for i in range(5):
    p=obs[i*16:i*16+3]
    u,v=pr(p); 
    if 0<=int(v)<480 and 0<=int(u)<640:
        c=readpng(env.render_state(state=obs.tolist(),label="ov_c")).astype(int)[int(round(v)),int(round(u))]
    else: c=None
    print('  cube%d %s -> (%.1f,%.1f) rgb=%s'%(i,np.round(p,3),u,v,c))
out=img.copy()
def dot(u,v,col,r=1):
    u,v=int(round(u)),int(round(v))
    out[max(0,v-r):v+r+1, max(0,u-r):u+r+1] = col
# island face top edge candidates
for y in np.arange(-1.3,1.31,0.05):
    u,v=pr([0.90,y,0.45])
    lbl = abs(y%0.5)<1e-6 or abs(abs(y%0.5)-0.5)<1e-6
    dot(u,v,[255,0,0], 2 if lbl else 0)
for y in np.arange(-1.3,1.31,0.1):
    u,v=pr([0.90,y,0.0]); dot(u,v,[0,0,255],0)
# handles
for y in (0.671,0.002,-0.666):
    for z in (0.0909,0.3115):
        u,v=pr([0.9227,y,z]); dot(u,v,[0,255,0],2)
writepng('crops/d_overlay.png', np.clip(out,0,255).astype(np.uint8))
big=np.repeat(np.repeat(np.clip(out,0,255).astype(np.uint8)[240:480,150:470],2,0),2,1)
writepng('crops/d_overlay_zoom.png', big)
print('wrote crops/d_overlay.png (full) and crops/d_overlay_zoom.png (origin 150,240 scale2)')
env.close()
