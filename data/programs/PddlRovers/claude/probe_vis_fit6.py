import numpy as np
D=[]
def add(pos,O,q,rows):
    for th,v in rows: D.append((np.array(pos),np.array(O),np.array(q),th,v))
add((1.416,1.396),(0.3665,2.056),(1.036,1.635),
    [(-3.14,0),(-2.62,0),(-2.09,0),(-1.57,1),(-1.05,1),(-0.52,1),(0.0,1),(0.52,1),(1.05,0),(1.57,0),(2.09,0),(2.62,0)])
add((1.471,1.328),(2.166,2.206),(1.753,1.685),
    [(-3.14,1),(-2.62,1),(-2.09,1),(-1.57,1),(-1.05,1),(-0.52,0),(0.0,0),(0.52,0),(1.05,0),(1.57,0),(2.09,0),(2.62,0)])
best=None
for a in np.arange(-0.30,0.301,0.005):
 for b in np.arange(-0.30,0.301,0.005):
  for w in np.arange(0.03,0.16,0.005):
    err=0
    for pos,O,q,th,v in D:
        e=np.array([np.cos(th),np.sin(th)]); n=np.array([-np.sin(th),np.cos(th)])
        s=pos+a*e+b*n
        d=np.linalg.norm(O-s); u=(O-s)/d
        wv=q-s; t=np.dot(wv,u); perp=abs(wv[0]*u[1]-wv[1]*u[0])
        pred = not (0<t<d and perp<w)
        err += (pred!=bool(v))
    if best is None or err<best[0]: best=(err,a,b,w)
print("best err=%d/%d  a=%.3f b=%.3f w=%.3f"%(best[0],len(D),best[1],best[2],best[3]))
# show all near-optimal
sols=[]
for a in np.arange(-0.30,0.301,0.01):
 for b in np.arange(-0.30,0.301,0.01):
  for w in np.arange(0.03,0.16,0.005):
    err=0
    for pos,O,q,th,v in D:
        e=np.array([np.cos(th),np.sin(th)]); n=np.array([-np.sin(th),np.cos(th)])
        s=pos+a*e+b*n
        d=np.linalg.norm(O-s); u=(O-s)/d
        wv=q-s; t=np.dot(wv,u); perp=abs(wv[0]*u[1]-wv[1]*u[0])
        pred = not (0<t<d and perp<w)
        err += (pred!=bool(v))
    if err<=best[0]+0: sols.append((err,round(a,3),round(b,3),round(w,3)))
print("solutions with err<=%d:"%best[0], sols[:40])
