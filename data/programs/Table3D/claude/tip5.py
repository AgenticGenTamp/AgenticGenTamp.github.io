import subprocess, numpy as np
def load(path):
    out = subprocess.run(['convert', path, '-depth','8','rgb:-'], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(360,640,3).astype(int)
for f in ['0000','0003','0006','0009','0011']:
    img=load('/sandbox/mcp_renders/state_custom_policy_seed0_%s.png'%f)
    lum=img.mean(2)
    m=np.zeros(lum.shape,bool); m[0:170,180:420]=True
    m&= lum<26
    ys,xs=np.nonzero(m)
    print('==',f,'n=%d bbox x[%d-%d] y[%d-%d] centroid(%.1f,%.1f)'%(len(ys),xs.min(),xs.max(),ys.min(),ys.max(),xs.mean(),ys.mean()))
    o=np.argsort(-ys)[:6]; print('   lowest:',[(int(xs[k]),int(ys[k])) for k in o])
    o=np.argsort(xs)[:6]; print('   leftmost:',[(int(xs[k]),int(ys[k])) for k in o])
    o=np.argsort(-xs)[:3]; print('   rightmost:',[(int(xs[k]),int(ys[k])) for k in o])
