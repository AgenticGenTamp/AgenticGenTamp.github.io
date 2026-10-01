import numpy as np, json, itertools
from pngtool import readpng
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
M=np.load('cam_M.npy')
def proj(X):
    X=np.atleast_2d(np.asarray(X,float)); h=np.c_[X,np.ones(len(X))]@M.T
    return h[:,:2]/h[:,2:3]
ref=readpng('mcp_renders/state_custom_cl_ref.png').astype(int)
H,W,_=ref.shape; YY,XX=np.mgrid[0:H,0:W]
def dm(p):
    im=readpng(p).astype(int); return np.abs(im-ref).sum(2)>18
def sil(c,h,win):
    x0,x1,y0,y1=win
    C=np.array(list(itertools.product(*[[c[i]-h[i],c[i]+h[i]] for i in range(3)])))
    uv=proj(C)
    try: hu=ConvexHull(uv)
    except Exception: return None
    A=hu.equations[:,:2]; b=hu.equations[:,2]
    xs=XX[y0:y1,x0:x1]; ys=YY[y0:y1,x0:x1]
    ins=np.ones(xs.shape,bool)
    for a,bb in zip(A,b): ins &= (a[0]*xs+a[1]*ys+bb)<=0.5
    return ins
LO=np.array([0.30,-1.15,0.02,0.10,0.05,0.03]); HI=np.array([0.95,1.15,0.50,0.34,0.30,0.20])
def loss(p,masks,win):
    p=np.clip(p,LO,HI)
    c=p[:3]; h=p[3:6]
    s0=sil(c,h,win)
    if s0 is None: return 5.
    tot=0
    for d,o in masks:
        s1=sil([c[0]+d,c[1],c[2]],h,win)
        if s1 is None: return 5.
        pm=s0^s1
        u=(pm|o).sum()
        tot+= 1-((pm&o).sum()/u if u else 0)
    return tot/len(masks)
res={}
for j in range(6):
    ma=dm('mcp_renders/state_custom_cl_d%da.png'%j); mb=dm('mcp_renders/state_custom_cl_d%db.png'%j)
    ys,xs=np.nonzero(mb)
    win=(max(0,xs.min()-30),min(W,xs.max()+30),max(0,ys.min()-30),min(H,ys.max()+30))
    masks=[(0.45,ma[win[2]:win[3],win[0]:win[1]]),(0.65,mb[win[2]:win[3],win[0]:win[1]])]
    best=(9,None)
    for yc in [-0.75,-0.4,-0.05,0.3,0.65]:
        for zc in [0.12,0.30]:
            for xc in [0.55,0.68]:
                p0=np.array([xc,yc,zc,0.20,0.15,0.09])
                r=minimize(loss,p0,args=(masks,win),method='Powell',options={'maxiter':4000,'xtol':1e-3,'ftol':1e-4})
                if r.fun<best[0]: best=(r.fun,np.clip(r.x,LO,HI))
    f,p=best
    res[j]=dict(iou=round(1-f,3),c=[round(v,3) for v in p[:3]],h=[round(v,3) for v in p[3:6]])
    print(j,'IoU %.3f'%(1-f),'closed_center',np.round(p[:3],3),'half',np.round(p[3:6],3),flush=True)
json.dump(res,open('fitbox2.json','w'),indent=1)
