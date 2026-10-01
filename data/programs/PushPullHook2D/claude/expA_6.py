import numpy as np, sys
from helper import Sim
from expA_lib import rel, robot_for
from expA_place import setup, cand
def place_any(s,Cl,dth,u,bars=('long','short'),ts=(0.3,0.6,0.9),s0=0.11,back=0.30,v=False):
    M=s.obs[20:22].copy(); u=np.asarray(u,float); u/=np.linalg.norm(u)
    for bar in bars:
        for t in ts:
            for hth,C in cand(u,bar,t,s0,M):
                Cs=C-u*back; Rp,rth=robot_for(Cl,dth,Cs,hth)
                if 0.16<Rp[0]<3.34 and 0.16<Rp[1]<1.10:
                    o=s.obs
                    a=s.goto(o[0],o[1],rth,0.2,1.0); b=s.goto(Rp[0],s.obs[1],rth,0.2,1.0); c=s.goto(Rp[0],Rp[1],rth,0.2,1.0)
                    err=np.linalg.norm(s.obs[:2]-Rp)
                    return dict(bar=bar,t=t,hth=hth,Rp=Rp,err=err,st=(a,b,c))
    return None
mode=sys.argv[1]
if mode=='term':
    for sd in [42,0,1,3,7]:
        s,Cl,dth=setup(sd)
        M=s.obs[20:22].copy(); T=s.obs[29:31].copy(); u=(T-M)/np.linalg.norm(T-M)
        r=place_any(s,Cl,dth,u)
        if r is None: print("sd%d NO PLACEMENT"%sd); s.close(); continue
        prev=None; d0=np.linalg.norm(M-T)
        for i in range(120):
            pd=np.linalg.norm(s.obs[20:22]-s.obs[29:31]); pM=s.obs[20:22].copy()
            s.step([u[0]*0.01,u[1]*0.01,0,0,1.0])
            nd=np.linalg.norm(s.obs[20:22]-s.obs[29:31])
            if s.term:
                print("sd%d bar=%s t=%.1f err=%.3f |MT| %.4f -> %.4f TERM at step %d (d0=%.3f)"%(sd,r['bar'],r['t'],r['err'],pd,nd,s.n,d0)); break
        else:
            print("sd%d bar=%s t=%.1f err=%.3f NOTERM final|MT|=%.4f d0=%.3f dM=%s"%(sd,r['bar'],r['t'],r['err'],nd,d0,np.round(s.obs[20:22]-M,3)))
        s.close()
elif mode=='corner':
    s,Cl,dth=setup(42); o=s.obs; M=o[20:22].copy()
    hth=np.pi/2; a=np.array([0.,-1.]); b=np.array([1.,0.])
    # nest button: dist 0.11 from each centerline, in quadrant (+x,-y) from corner
    for name,Cdes,mv in [("nest",M-np.array([0.11,-0.11]),None)]:
        pass
    Cdes=np.array([M[0]-0.11, M[1]+0.11]) + np.array([0.30,-0.30])/np.sqrt(2)*1.0  # start retracted along -(1,-1)
    Rp,rth=robot_for(Cl,dth,Cdes,hth)
    print("corner: robot target",np.round(Rp,3))
    s.goto(o[0],o[1],rth,0.2,1.0); s.goto(Rp[0],s.obs[1],rth,0.2,1.0); s.goto(Rp[0],Rp[1],rth,0.2,1.0)
    print(" placed C=",np.round(s.obs[9:11],4)," M=",np.round(s.obs[20:22],4)," relM-C=",np.round(s.obs[20:22]-s.obs[9:11],4),flush=True)
    d=np.array([1.,-1.])/np.sqrt(2)*0.01
    C0=s.obs[9:11].copy(); M0=s.obs[20:22].copy()
    for i in range(60):
        s.step([d[0],d[1],0,0,1.0])
        if s.term: break
    print(" diag push: dC=%s dM=%s rel=%s term=%s"%(np.round(s.obs[9:11]-C0,4),np.round(s.obs[20:22]-M0,4),np.round(s.obs[20:22]-s.obs[9:11],4),s.term),flush=True)
    # now try pure +x and pure -y from nest
    for v,lab in [((0.01,0),"+x"),((0,-0.01),"-y"),((-0.01,0.01),"reverse(-x,+y) pull?")]:
        C0=s.obs[9:11].copy(); M0=s.obs[20:22].copy()
        for i in range(20):
            s.step([v[0],v[1],0,0,1.0])
            if s.term: break
        print(" %s: dC=%s dM=%s rel=%s"%(lab,np.round(s.obs[9:11]-C0,4),np.round(s.obs[20:22]-M0,4),np.round(s.obs[20:22]-s.obs[9:11],4)),flush=True)
    s.close()
elif mode=='base':
    for sd in [1,42]:
        s=Sim(sd); o=s.obs; M=o[20:22].copy()
        print("sd%d M=%s"%(sd,np.round(M,3)),flush=True)
        k=s.goto(o[0],o[1],np.pi/2,0.2,0.0); k2=s.goto(M[0],min(1.13,M[1]-0.2),np.pi/2,0.2,0.0)
        print("  robot=%s (goto %d,%d)"%(np.round(s.obs[:3],3),k,k2),flush=True)
        M0=s.obs[20:22].copy()
        for i in range(30): s.step([0,0.02,0,0,0.0])
        print("  after +y pushes robot=%s dM=%s"%(np.round(s.obs[:2],3),np.round(s.obs[20:22]-M0,4)),flush=True)
        s.close()
