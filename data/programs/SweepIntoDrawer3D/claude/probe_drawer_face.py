import numpy as np
from env_client import make_env
from pngtool import readpng
M=np.load('cam_M.npy')
def pr(P):
    h=M@np.r_[np.asarray(P,float),1.0]; return np.array([h[0]/h[2],h[1]/h[2]])
env=make_env(); obs,_=env.reset(seed=0)
st=obs.copy(); st[125]=3.0; st[126]=-2.0; st[147:163]=0.0
img=readpng(env.render_state(state=st.tolist(),label="face")).astype(int)
def px(u,v):
    u=int(round(u)); v=int(round(v))
    if 0<=v<480 and 0<=u<640: return img[v,u]
    return np.array([-1,-1,-1])
# 1) y-extent of front face: sample at z=0.25 (mid face), x=0.90
print('y sweep at x=0.90 z=0.25 (rgb):')
prev=None
for y in np.arange(-1.40,1.41,0.02):
    c=px(*pr([0.90,y,0.25])); s='face' if 60<c[0]<200 else ('bg' if c[0]>230 else 'other')
    if s!=prev: print('   y=%.2f  rgb=%s  %s'%(y,c,s)); prev=s
# 2) z of top front edge: vertical scan in world at several y
print('\nz sweep (x=0.90) rgb transitions:')
for y in (-0.6,0.0,0.6):
    prev=None; 
    for z in np.arange(0.55,0.10,-0.005):
        c=px(*pr([0.90,y,z])); s=str(c)
        if prev is None or abs(int(c[0])-prev)>8:
            print('   y=%.1f z=%.3f rgb=%s'%(y,z,c)); prev=int(c[0])
# 3) front-face x: scan x at fixed y,z on the top surface->front transition
print('\nx sweep at z=0.452 (just above face, on counter top) y=0.0:')
prev=None
for x in np.arange(1.05,0.70,-0.01):
    c=px(*pr([x,0.0,0.452]))
    print('   x=%.2f rgb=%s'%(x,c))
env.close()
