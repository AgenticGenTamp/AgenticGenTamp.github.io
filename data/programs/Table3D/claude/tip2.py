import subprocess, numpy as np
def load(path):
    out = subprocess.run(['convert', path, '-depth','8','rgb:-'], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(360,640,3).astype(int)
base=load('/sandbox/mcp_renders/state_custom_policy_seed0_0000.png')
for f in ['0003','0006','0009','0011']:
    img=load('/sandbox/mcp_renders/state_custom_policy_seed0_%s.png'%f)
    d=np.abs(img-base).sum(2)
    changed=d>25
    R,G,B=img[:,:,0],img[:,:,1],img[:,:,2]
    lum=(R+G+B)/3.0
    grip=changed&(lum<45)
    link=changed&(lum>90)
    print('==',f)
    for name,m in [('gripper(dark,changed)',grip),('links(light,changed)',link)]:
        ys,xs=np.nonzero(m)
        if len(ys)==0: print('  ',name,'none'); continue
        print('  %s n=%d bbox x[%d-%d] y[%d-%d] centroid(%.1f,%.1f)'%(name,len(ys),xs.min(),xs.max(),ys.min(),ys.max(),xs.mean(),ys.mean()))
    ys,xs=np.nonzero(grip)
    # tip = lowest points of gripper
    order=np.argsort(-ys)[:15]
    print('   lowest gripper px:', [(int(xs[k]),int(ys[k])) for k in order])
    k=np.argmin(xs); print('   leftmost gripper px:',int(xs[k]),int(ys[k]))
