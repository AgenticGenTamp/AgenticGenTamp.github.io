import numpy as np, json, itertools, sys
from pngtool import readpng
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
M=np.load('cam_M.npy')
def proj(X):
    X=np.atleast_2d(np.asarray(X,float)); h=np.c_[X,np.ones(len(X))]@M.T
    return h[:,:2]/h[:,2:3]
ref=readpng('mcp_renders/state_seed0_seed0_init.png').astype(int)
H,W,_=ref.shape
YY,XX=np.mgrid[0:H,0:W]
def diffmask(name,th=18):
    im=readpng('mcp_renders/state_custom_%s.png'%name).astype(int)
    return (np.abs(im-ref).sum(2)>th)
def sil(c,h,win):
    x0,x1,y0,y1=win
    corners=np.array(list(itertools.product(*[[c[i]-h[i],c[i]+h[i]] for i in range(3)])))
    uv=proj(corners)
    try: hull=ConvexHull(uv)
    except Exception: return None
    A=hull.equations[:,:2]; b=hull.equations[:,2]
    xs=XX[y0:y1,x0:x1]; ys=YY[y0:y1,x0:x1]
    inside=np.ones(xs.shape,bool)
    for a,bb in zip(A,b):
        inside &= (a[0]*xs+a[1]*ys+bb)<=0.5
    return inside
def predmask(c,h,win,d=0.45):
    s0=sil(c,h,win); s1=sil([c[0]+d,c[1],c[2]],h,win)
    if s0 is None or s1 is None: return None
    return s0^s1, s0|s1
def loss(p,obs,win,mode='sym'):
    c=p[:3]; h=np.abs(p[3:6])
    r=predmask(c,h,win)
    if r is None: return 5.0
    sym,uni=r
    pm = sym if mode=='sym' else uni
    o=obs
    inter=(pm&o).sum(); union=(pm|o).sum()
    if union==0: return 5.0
    return 1.0-inter/union
names=['drawer0','drawer1','drawer2','drawer3','drawer4','drawer5']
out={}
for k,n in enumerate(names):
    o=diffmask(n)
    ys,xs=np.nonzero(o)
    win=(max(0,xs.min()-40),min(W,xs.max()+40),max(0,ys.min()-40),min(H,ys.max()+40))
    ow=o[win[2]:win[3],win[0]:win[1]]
    best=(9,None,None)
    for mode in ['sym','uni']:
      for xc in [0.55,0.65,0.72]:
        for yc in [-0.7,-0.35,0.0,0.35,0.7]:
            for zc in [0.15,0.25,0.35]:
                for hy in [0.12,0.2]:
                    p0=np.array([xc,yc,zc,0.15,hy,0.08])
                    r=minimize(loss,p0,args=(ow,win,mode),method='Powell',
                               options={'maxiter':3000,'xtol':1e-3,'ftol':1e-3})
                    if r.fun<best[0]: best=(r.fun,r.x.copy(),mode)
    f,p,mode=best
    c=p[:3]; h=np.abs(p[3:6])
    out[n]=dict(iou=1-f,mode=mode,center_closed=list(np.round(c,3)),half=list(np.round(h,3)))
    print(n,'IoU=%.3f'%(1-f),mode,'closed center',np.round(c,3),'half',np.round(h,3),flush=True)
json.dump(out,open('fitbox.json','w'),indent=1)
