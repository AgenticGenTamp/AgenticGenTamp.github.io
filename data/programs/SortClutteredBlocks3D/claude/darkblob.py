import numpy as np,sys
from pngread import read_png
im=read_png(sys.argv[1]).astype(int)
h,w,_=im.shape
mx=im.max(2); mn=im.min(2)
dark=(mx<90)&((mx-mn)<40)
ys,xs=np.nonzero(dark)
# restrict to lower half of image (near cubes)
m=(ys>320)&(ys<430)
print('dark pixels rows 320-430:', m.sum())
if m.sum():
    print('row range',ys[m].min(),ys[m].max(),'col range',xs[m].min(),xs[m].max(),'centroid',round(ys[m].mean(),1),round(xs[m].mean(),1))
    # lowest rows
    for r in range(int(ys[m].min()), int(ys[m].max())+1, 10):
        sel=(ys==r)&dark[ys,xs] if False else (ys==r)
        cols=xs[(ys==r)]
        if len(cols): print(' row',r,'cols',cols.min(),cols.max(),len(cols))
