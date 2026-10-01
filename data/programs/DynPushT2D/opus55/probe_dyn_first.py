import numpy as np
exec(open('probe_dyn_fit.py').read().split("rows=[]")[0])
moved=np.abs(D[:,10:13]-D[:,3:6]).max(1)>1e-8
out=[]
for i in range(1,len(D)):
    r=D[i]
    if not moved[i] or (moved[i-1] and D[i-1,1]==r[1]): continue
    R=rot(r[5]).T; q=R@(r[6:8]-r[3:5]); u=R@r[8:10]
    p,n,d=closest(q,*r[15:18]); g=d-0.1; un=-(u@n)
    # post-step robot relative to post-step block
    R2=rot(r[12]).T; q2=R2@(r[13:15]-r[10:13][:2]); p2,n2,d2=closest(q2,*r[15:18])
    out.append([r[18],g,un,un-g,d2-0.1])
O=np.array(out)
print('first-contact steps: v, pre-gap, u_n, penetration-if-static(u_n-g), post-step gap')
for o in O[np.argsort(O[:,3])][::max(1,len(O)//15)]: print(o.round(4))
