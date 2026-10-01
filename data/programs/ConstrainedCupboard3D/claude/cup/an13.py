import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
a=load('cup/s0.ppm').astype(int)
R,G,B=a[...,0],a[...,1],a[...,2]
# rods: reddish brown, brighter red vs blue
m=((R-B)>=35)&(R>110)&(R<200)
m[:150]=False
ys,xs=np.nonzero(m)
print('n',m.sum())
# cluster crudely
from collections import deque
lab=-np.ones(a.shape[:2],int); c=0
pts=list(zip(ys,xs)); S=set(pts)
for p in pts:
    if lab[p]>=0: continue
    q=deque([p]); lab[p]=c
    while q:
        y,x=q.popleft()
        for dy in(-2,-1,0,1,2):
            for dx in(-2,-1,0,1,2):
                n=(y+dy,x+dx)
                if n in S and lab[n]<0: lab[n]=c; q.append(n)
    c+=1
for i in range(c):
    yy,xx=np.nonzero(lab==i)
    if len(yy)>80: print('blob',i,'n',len(yy),'centroid y%.0f x%.0f'%(yy.mean(),xx.mean()),'ybox',yy.min(),yy.max(),'xbox',xx.min(),xx.max())
