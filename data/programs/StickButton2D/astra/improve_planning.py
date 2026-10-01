p='approach.py'
s=open(p).read()
a=s.index('    def direct_goal(');b=s.index('    def get_action(',a)
s=s[:a]+'''    def direct_goal(self,r,b):
        t=np.linspace(-math.pi,math.pi,97)
        p=b-.2*np.stack([np.cos(t),np.sin(t)],axis=1)
        ok=(p[:,0]>=.101)&(p[:,0]<=3.399)&(p[:,1]>=.101)&(p[:,1]<=1.149)
        dt=(t-r[2]+math.pi)%TAU-math.pi
        cost=np.maximum(np.max(abs(p-r[:2]),axis=1)/.05,abs(dt)/.196)
        cost[~ok]=np.inf
        k=np.argmin(cost)
        return None if not np.isfinite(cost[k]) else np.r_[p[k],t[k],.2]
    def held_goal(self,r,st,b):
        sx,sy,alpha,w,h=st
        u0=np.array([math.cos(alpha),math.sin(alpha)])
        v0=np.array([-math.sin(alpha),math.cos(alpha)])
        off=np.array([sx,sy])+w/2*u0-r[:2]
        dt=np.linspace(-math.pi,math.pi,145)
        c,ss=np.cos(dt),np.sin(dt)
        def rot(v):
            return np.stack([c*v[0]-ss*v[1],ss*v[0]+c*v[1]],axis=1)
        q=rot(off);v=rot(v0);u=rot(u0);t=r[2]+dt
        grip=.2*np.stack([np.cos(t),np.sin(t)],axis=1)
        ext=np.stack([.005*abs(np.cos(t))+.035*abs(np.sin(t)),.005*abs(np.sin(t))+.035*abs(np.cos(t))],axis=1)
        lo=np.maximum([.101,.101],ext-grip+.001)
        hi=np.minimum([3.399,1.149],np.stack([3.499-ext[:,0]-grip[:,0],np.full(len(t),1.149)],axis=1))
        corners=np.stack([q-w/2*u,q+w/2*u,q-w/2*u+h*v,q+w/2*u+h*v],axis=1)
        lo=np.maximum(lo,-corners.min(axis=1)+.001)
        hi=np.minimum(hi,np.array([3.499,2.499])-corners.max(axis=1))
        lmin=np.full(len(t),-.027);lmax=np.full(len(t),h+.027)
        for k in range(2):
            flat=abs(v[:,k])<1e-8
            safe=np.where(flat,1.,v[:,k])
            a=(b[k]-q[:,k]-hi[:,k])/safe;bb=(b[k]-q[:,k]-lo[:,k])/safe
            lmin=np.maximum(lmin,np.where(flat,-np.inf,np.minimum(a,bb)))
            lmax=np.minimum(lmax,np.where(flat,np.inf,np.maximum(a,bb)))
            lmax[flat&((b[k]-q[:,k]<lo[:,k])|(b[k]-q[:,k]>hi[:,k]))]=-100
        lengths=np.clip(np.sum((b-q-r[:2])*v,axis=1),lmin,np.maximum(lmin,lmax))
        p=b-q-lengths[:,None]*v
        cost=np.maximum(np.max(abs(p-r[:2]),axis=1)/.05,abs(dt)/.196)+.02*abs(dt)
        cost[(lmin>lmax)|np.any(lo>hi,axis=1)]=np.inf
        k=np.argmin(cost)
        return None if not np.isfinite(cost[k]) else np.r_[p[k],wrap(t[k]),.2]
''' +s[b:]
open(p,'w').write(s)
