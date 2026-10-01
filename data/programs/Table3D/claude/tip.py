import subprocess, numpy as np
def load(path):
    out = subprocess.run(['convert', path, '-depth','8','rgb:-'], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(360,640,3).astype(int)
for f in ['0000','0003','0006','0009','0011']:
    img=load('/sandbox/mcp_renders/state_custom_policy_seed0_%s.png'%f)
    R,G,B=img[:,:,0],img[:,:,1],img[:,:,2]
    dark=(R<75)&(G<75)&(B<75)
    m=np.zeros_like(dark); m[0:120,200:360]=True
    dark=dark&m
    ys,xs=np.nonzero(dark)
    print('--',f,'dark gripper px n=%d bbox x[%d-%d] y[%d-%d] centroid (%.1f,%.1f)'%(len(ys),xs.min(),xs.max(),ys.min(),ys.max(),xs.mean(),ys.mean()))
    # lowest point (max y) and leftmost
    i=np.argmax(ys); print('   lowest dark px:',xs[i],ys[i])
    j=np.argmin(xs); print('   leftmost dark px:',xs[j],ys[j])
    # bottom row extent
    for yy in range(ys.max(),ys.max()-4,-1):
        row=xs[ys==yy]
        if len(row): print('    y=%d x %d..%d'%(yy,row.min(),row.max()))
