import subprocess, numpy as np, glob, os
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
a=load('mcp_renders/state_custom_policy_seed0_0000_1.png'); b=load('mcp_renders/state_custom_policy_seed0_0009_1.png')
for nm,im in (('f0',a),('f9',b)):
    R,G,B=im[...,0],im[...,1],im[...,2]
    p=(R>35)&(G<45)&(B>35)&(np.abs(R-B)<25)&(R>2*G+10)
    ys,xs=np.nonzero(p)
    print(nm,'total purple px',p.sum())
    # column profile in right cluster
    for x0,x1 in ((265,290),(350,400)):
        sel=(xs>=x0)&(xs<=x1)
        print('  region',x0,x1,'cols:',[(x,int((xs[sel]==x).sum())) for x in range(x0,x1+1) if (xs[sel]==x).sum()>0][:60])
print('diff purple px:', int((( (a[...,0]>35)&(a[...,1]<45)&(a[...,2]>35) ) != ( (b[...,0]>35)&(b[...,1]<45)&(b[...,2]>35) )).sum()))
