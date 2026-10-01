import zlib, struct, math, sys
import numpy as np
S=100
def png(path, img):
    h,w,_=img.shape
    raw=b''.join(b'\x00'+img[y].tobytes() for y in range(h))
    def chunk(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    open(path,'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
def poly_fill(img, pts, col):
    H,W,_=img.shape
    pts=np.array(pts)
    x0,y0=np.floor(pts.min(0)).astype(int); x1,y1=np.ceil(pts.max(0)).astype(int)
    x0=max(x0,0);y0=max(y0,0);x1=min(x1,W-1);y1=min(y1,H-1)
    if x1<x0 or y1<y0: return
    yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
    inside=np.ones_like(xx,bool); n=len(pts)
    sgn=None
    for i in range(n):
        ax,ay=pts[i]; bx,by=pts[(i+1)%n]
        c=(bx-ax)*(yy-ay)-(by-ay)*(xx-ax)
        inside &= (c>=0) if (sgn:=1) else True
    if not inside.any():
        inside=np.ones_like(xx,bool)
        for i in range(n):
            ax,ay=pts[i]; bx,by=pts[(i+1)%n]
            inside &= ((bx-ax)*(yy-ay)-(by-ay)*(xx-ax))<=0
    img[y0:y1+1,x0:x1+1][inside]=col
def P(x,y,H): return (x*S, H-y*S)
def rect(img,cx,cy,th,w,h,col,H):
    c,s=math.cos(th),math.sin(th); pts=[]
    for dx,dy in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]:
        pts.append(P(cx+c*dx-s*dy, cy+s*dx+c*dy,H))
    poly_fill(img,pts,col)
def circle(img,cx,cy,r,col,H):
    pts=[P(cx+r*math.cos(a),cy+r*math.sin(a),H) for a in np.linspace(0,2*math.pi,24,endpoint=False)]
    poly_fill(img,pts,col)
def render(obs, path):
    W,H=int(3.5*S),int(3.0*S)
    img=np.full((H,W,3),255,np.uint8)
    rect(img,1.75,0.75,0,0.1,1.5,(30,30,30),H)
    for n in obs.get_object_names():
        o=obs.get_object_from_name(n); g=lambda f: obs.get(o,f)
        if n.startswith('small'):
            if 'radius' in [f for f in obs.get_object_from_name(n).type.feature_names] if hasattr(o.type,'feature_names') else False: pass
            try: circle(img,g('x'),g('y'),g('radius'),(255,165,0),H)
            except Exception: rect(img,g('x'),g('y'),g('theta'),g('size'),g('size'),(128,0,128),H)
    o=obs.get_object_from_name('hook'); g=lambda f: obs.get(o,f)
    x,y,th=g('x'),g('y'),g('theta'); u=(math.cos(th),math.sin(th)); v=(-math.sin(th),math.cos(th))
    # vertical bar along -u from corner, horizontal bar along -v
    L=0.5;wd=0.05
    cx=x-u[0]*L/2-v[0]*wd/2; cy=y-u[1]*L/2-v[1]*wd/2; rect(img,cx,cy,th,L,wd,(120,60,20),H)
    cx=x-v[0]*L/2-u[0]*wd/2; cy=y-v[1]*L/2-u[1]*wd/2; rect(img,cx,cy,th+math.pi/2,L,wd,(160,80,30),H)
    o=obs.get_object_from_name('robot'); g=lambda f: obs.get(o,f)
    x,y,th,aj,gap=g('x'),g('y'),g('theta'),g('arm_joint'),g('finger_gap')
    circle(img,x,y,0.2,(150,0,150),H)
    d=(math.cos(th),math.sin(th)); n=(-d[1],d[0])
    rect(img,x+d[0]*aj/2,y+d[1]*aj/2,th,aj,0.03,(80,0,80),H)
    rect(img,x+d[0]*(aj+0.02),y+d[1]*(aj+0.02),th,0.04,0.25,(80,0,80),H)
    for sgn in (1,-1):
        rect(img,x+d[0]*(aj+0.08)+sgn*n[0]*gap/2,y+d[1]*(aj+0.08)+sgn*n[1]*gap/2,th,0.12,0.04,(80,0,80),H)
    png(path,img)
