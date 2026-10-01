import sys, numpy as np
from env_client import make_env
import approach as A
from collide import Obstacles, link_points, point_box_dist, table_clear, _seg_samples
from planner import TABLE
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=A.GeneratedApproach(env.action_space, env.observation_space, None)
ap.reset(obs,info)
def yaw(b): x,y,z,w=b[3:7]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
g0=ap._block('green0'); gy=yaw(g0); c,s=np.cos(gy),np.sin(gy)
def pen_box(u,v,hu,hv): return (g0[0]+c*u-s*v, g0[1]+s*u+c*v, 0.8, hu, hv, 0.07, gy)
pen=[pen_box(0.08,0.15,0.08,0.01),pen_box(0.08,-0.15,0.08,0.01),pen_box(0.16,0,0.01,0.16)]
def report(b,q):
    sh,el,wr,tool,ax=link_points(b,q)
    names=['blocker','green0']
    boxes=[]
    for n in names:
        bb=ap._block(n); boxes.append((n,(bb[0],bb[1],bb[2],0.035,0.035,0.07,yaw(bb))))
    boxes+= [('penL',pen[0]),('penR',pen[1]),('penB',pen[2])]
    out=[]
    for name,a,cc in [('upper',sh,el),('fore',el,wr),('grip',wr,tool)]:
        pts=_seg_samples(a,cc,8)
        dt=min(table_clear(p,TABLE,0) for p in pts)
        dd={n:round(min(point_box_dist(p,bx) for p in pts),3) for n,bx in boxes}
        out.append((name,'tbl',round(dt,3),dd, 'zmin',round(min(p[2] for p in pts),3)))
    return out
for t in range(1000):
    a=ap.get_action(obs)
    b0=ap._base(); q0=ap._q()
    obs,r,term,trunc,_=env.step(a)
    if term: print('done',t); break
    if np.abs(np.r_[ap._robot and 0, 0])[0]==0 and np.allclose(np.r_[b0,q0], np.r_[np.array([float(obs.get(ap._r,f)) for f in ['base_x','base_y','base_rot']+A.JN])]) and np.abs(a[:10]).max()>1e-4:
        print('step',t,'BLOCK cmd',np.round(a[:10],2))
        for l in report(b0+a[:3], q0+a[3:10]): print('   ',l)
    if t>300: break
