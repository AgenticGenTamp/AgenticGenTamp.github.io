from drawer_util import *
r=R()
bd=r.base.copy(); bd[0]=1.6
r.goto(r.q,40,base_d=bd); print('base',r.base)
y=-0.08
for z in [0.44,0.40,0.36,0.32,0.28,0.24,0.20,0.16,0.12,0.08]:
    q,e=r.ik([1.15,y,z],RA); err0=r.goto(q,80)
    x=1.15
    while x>0.6:
        x-=0.01
        q,e=r.ik([x,y,z],RA); err=r.goto(q,15)
        if err>0.02: break
    print('z',z,'err0 %.3f ikerr %.4f'%(err0,e),'contact cmd x %.2f'%x,'tool',r.tool()[:3,3],'err %.3f'%err,'dr',r.obs[103:109])
    q,e=r.ik([1.15,y,z],RA); r.goto(q,40)
print(set(r.rews), r.n)
