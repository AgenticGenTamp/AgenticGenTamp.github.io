import numpy as np
def rect_corners(x,y,th,w,h,mode='corner'):
    c,s=np.cos(th),np.sin(th); u=np.array([c,s]); v=np.array([-s,c])
    if mode=='corner': o=np.array([x,y])
    else: o=np.array([x,y])-u*w/2-v*h/2
    return np.array([o,o+u*w,o+u*w+v*h,o+v*h])
def rect_centered(cx,cy,th,w,h): return rect_corners(cx,cy,th,w,h,'center')
def poly_poly(P,Q,eps=0.0):
    for poly in (P,Q):
        for i in range(len(poly)):
            e=poly[(i+1)%len(poly)]-poly[i]; n=np.array([-e[1],e[0]]); n/=np.linalg.norm(n)
            a=P@n; b=Q@n
            if a.max()<b.min()-eps or b.max()<a.min()-eps: return False
    return True
def seg_dist(p,a,b):
    ab=b-a; t=np.clip(np.dot(p-a,ab)/np.dot(ab,ab),0,1); return np.linalg.norm(p-(a+t*ab))
def inside(p,P):
    sg=[np.cross(P[(i+1)%4]-P[i],p-P[i]) for i in range(4)]
    return all(x>=0 for x in sg) or all(x<=0 for x in sg)
def circ_poly(c,r,P):
    if inside(c,P): return True
    return min(seg_dist(c,P[i],P[(i+1)%4]) for i in range(4))<r
