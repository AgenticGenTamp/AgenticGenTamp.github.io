import subprocess, numpy as np
from scipy import ndimage
def load(path):
    out = subprocess.run(['convert', path, '-depth','8','rgb:-'], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(360,640,3).astype(int)
for f in ['0000','0003','0006','0009','0011']:
    img=load('/sandbox/mcp_renders/state_custom_policy_seed0_%s.png'%f)
    lum=img.mean(2)
    m=np.zeros(lum.shape,bool); m[0:170,180:420]=True
    m&= lum<26
    lab,n=ndimage.label(m, structure=np.ones((3,3)))
    print('==',f)
    for i in range(1,n+1):
        ys,xs=np.nonzero(lab==i)
        if len(ys)<30: continue
        w=xs.max()-xs.min()+1
        print('  n=%4d x[%d-%d] y[%d-%d] cen(%.1f,%.1f) w=%d'%(len(ys),xs.min(),xs.max(),ys.min(),ys.max(),xs.mean(),ys.mean(),w))

print('--- fingertip (lowest gripper px) per frame ---')
for f in ['0000','0003','0006','0009','0011']:
    img=load('/sandbox/mcp_renders/state_custom_policy_seed0_%s.png'%f)
    lum=img.mean(2)
    m=np.zeros(lum.shape,bool); m[0:170,180:420]=True
    m&= lum<26
    lab,n=ndimage.label(m, structure=np.ones((3,3)))
    sizes=ndimage.sum(m,lab,range(1,n+1))
    i=int(np.argmax(sizes))+1
    ys,xs=np.nonzero(lab==i)
    ymax=ys.max(); row=xs[ys>=ymax-1]
    print(f,'bottom y=%d x %d..%d ; xmin=%d(at y=%d) ; xmax=%d'%(ymax,row.min(),row.max(),xs.min(),ys[np.argmin(xs)],xs.max()))
