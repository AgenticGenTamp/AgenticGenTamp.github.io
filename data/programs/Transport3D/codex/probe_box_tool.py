"""Use the reliably grasped box as a kinematic pusher for one cube."""
import math
import sys
import numpy as np
from env_client import make_env

Q = [
 [.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593],
 [4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916],
 [4.170779777,1.703020948,-2.855671258,-.627929854,4.26973606,1.31947716,7.283258434]]
OFF=[[-.799987478,-.347800459,2.104792961],[.126212298,-.083534270,-3.139703362],[-.428185141,.054943428,2.978131848]]

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xyz(s,n): return np.array([v(s,n,'pose_'+c) for c in 'xyz'])
def command(e,s,b,q,grip,n=1):
    for _ in range(n):
        a=np.zeros(11,np.float32)
        cur=[v(s,'robot','pos_base_x'),v(s,'robot','pos_base_y'),v(s,'robot','pos_base_rot')]
        cq=[v(s,'robot','joint_%d'%i) for i in range(1,8)]
        a[:3]=np.clip(np.asarray(b)-cur,-.2,.2);a[3:10]=np.clip(np.asarray(q)-cq,-.2,.2);a[10]=grip
        s,r,t,tr,_=e.step(a)
        if t or tr:return s,t
    return s,False

def run(seed=1, mode=0, probe_joint=0, probe_sign=1):
    e=make_env();s,_=e.reset(seed=seed,options={'object_count':1})
    tx,ty=xyz(s,'box0')[:2]; c0=xyz(s,'cube0').copy()
    for q,o in zip(Q,OFF):
        s,_=command(e,s,[tx+o[0],ty+o[1],o[2]],q,1,35)
        s,_=command(e,s,[tx+o[0],ty+o[1],o[2]],q,-1,2)
    print('grasp',v(s,'robot','grasp_active'),xyz(s,'box0'),c0)
    q=np.array([v(s,'robot','joint_%d'%i) for i in range(1,8)])
    # Work along the cube-to-table direction. Put the box just behind the cube.
    c=xyz(s,'cube0'); u=np.array([.6-c[0],-c[1]]);u/=np.linalg.norm(u)
    if mode == 9: u=np.array([.93571114,-.35276714])
    side=c[:2]-u*(.18 if mode!=1 else .28)
    for k in range(15):
        p=xyz(s,'box0'); b=[v(s,'robot','pos_base_x')+side[0]-p[0],v(s,'robot','pos_base_y')+side[1]-p[1],v(s,'robot','pos_base_rot')]
        s,t=command(e,s,b,q,-1)
    print('staged box/cube',xyz(s,'box0'),xyz(s,'cube0'),'u',u)
    # Sweep box toward table while lifting its center. Variants adjust lift rate.
    dq=(-.10 if mode in (0,2,3,4,5,6,7,8,9) else -.04)
    lift_steps = 10 if mode in (3,4,5,6,7,8,9) else 30
    for k in range(lift_steps):
        p=xyz(s,'box0'); c=xyz(s,'cube0')
        dest=np.array([.6,0.])
        step=np.clip(dest-p[:2],-.08,.08)
        a=np.zeros(11,np.float32);a[:2]=step;a[4]=dq;a[10]=-1
        s,r,t,tr,_=e.step(a)
        if k%2==0:print('sweep',k,'b',np.round(xyz(s,'box0'),3),'c',np.round(xyz(s,'cube0'),3),t)
        if t or tr:break
    if mode in (3,4,5,6,7,8,9):
        # At this point the cube bottom is just above the tabletop. Continue
        # pushing horizontally with the box, without changing arm height.
        for k in range(35):
            c=xyz(s,'cube0'); p=xyz(s,'box0')
            d=np.array([.60-c[0], -c[1]])
            a=np.zeros(11,np.float32);a[:2]=np.clip(d,-.10,.10);a[10]=-1
            s,r,t,tr,_=e.step(a)
            if k%2==0: print('carry',k,'b',np.round(xyz(s,'box0'),3),'c',np.round(xyz(s,'cube0'),3),t)
            if t or tr: break
        if mode in (6,7,8,9):
            for k in range(12):
                c=xyz(s,'cube0')
                a=np.zeros(11,np.float32);a[:2]=np.clip([.6-c[0],-c[1]],-.08,.08);a[4]=.10;a[10]=-1
                s,r,t,tr,_=e.step(a)
                print('lowerpair',k,np.round(xyz(s,'box0'),3),np.round(xyz(s,'cube0'),3),t)
                if t or tr: break
            if mode in (7,8,9):
                for k in range(8):
                    c=xyz(s,'cube0')
                    a=np.zeros(11,np.float32);a[:2]=np.clip([.6-c[0],-c[1]],-.08,.08)
                    a[3+probe_joint]=.10*probe_sign;a[10]=-1
                    s,r,t,tr,_=e.step(a)
                    print('jointprobe',probe_joint,probe_sign,k,np.round(xyz(s,'box0'),3),np.round(xyz(s,'cube0'),3),t)
                    if t or tr: break
                if mode in (8,9):
                    for k in range(18):
                        b=xyz(s,'box0')
                        base=[v(s,'robot','pos_base_x')+.65-b[0],v(s,'robot','pos_base_y')+.25-b[1],v(s,'robot','pos_base_rot')]
                        s,t=command(e,s,base,q,-1)
                        print('boxplace',k,np.round(xyz(s,'box0'),3),np.round(xyz(s,'cube0'),3),t)
                    for k in range(4):
                        a=np.zeros(11,np.float32);a[10]=1;s,r,t,tr,_=e.step(a)
                        print('releasebox',k,t,np.round(xyz(s,'box0'),3),np.round(xyz(s,'cube0'),3))
                        if t or tr:break
        elif mode == 5:
            # Move opposite the box->cube contact normal, then lift the box
            # clear, center it above the stationary cube, and press downward.
            for k in range(5):
                a=np.zeros(11,np.float32);a[0]=-.12;a[1]=.12;a[10]=-1
                s,r,t,tr,_=e.step(a)
            print('separate',np.round(xyz(s,'box0'),3),np.round(xyz(s,'cube0'),3),t)
            for k in range(7):
                a=np.zeros(11,np.float32);a[4]=-.10;a[10]=-1
                s=e.step(a)[0]
            for k in range(10):
                b=xyz(s,'box0');c=xyz(s,'cube0')
                a=np.zeros(11,np.float32);a[:2]=np.clip(c[:2]-b[:2],-.12,.12);a[10]=-1
                s=e.step(a)[0]
            print('above',np.round(xyz(s,'box0'),3),np.round(xyz(s,'cube0'),3))
            for k in range(20):
                b=xyz(s,'box0');c=xyz(s,'cube0')
                a=np.zeros(11,np.float32);a[:2]=np.clip(c[:2]-b[:2],-.08,.08);a[4]=.10;a[10]=-1
                s,r,t,tr,_=e.step(a)
                print('press',k,np.round(xyz(s,'box0'),3),np.round(xyz(s,'cube0'),3),t)
                if t or tr:break
        else:
            for k in range(8):
                a=np.zeros(11,np.float32);a[1]=-.15;a[10]=-1
                s,r,t,tr,_=e.step(a)
            print('separate',np.round(xyz(s,'box0'),3),np.round(xyz(s,'cube0'),3),t)
    e.close()

if __name__=='__main__':run(int(sys.argv[1]) if len(sys.argv)>1 else 1,int(sys.argv[2]) if len(sys.argv)>2 else 0,int(sys.argv[3]) if len(sys.argv)>3 else 0,int(sys.argv[4]) if len(sys.argv)>4 else 1)
