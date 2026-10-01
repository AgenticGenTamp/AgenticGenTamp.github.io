import numpy as np
import kin
from plib2 import *
CUB_D=np.array([-0.015,0.0,0.01])   # gripper-frame recentering delta

def part_info(obs,name):
    o=obs.get_object_from_name(name); f=obs.type_features[o.type]
    return o,{k:obs.get(o,k) for k in f}

def yaw_of(obs,name):
    o=obs.get_object_from_name(name)
    qz=obs.get(o,'pose_qz'); qw=obs.get(o,'pose_qw')
    return 2*np.arctan2(qz,qw)

def target_for(obs,name,R,c=np.zeros(3)):
    """world linpos target to grasp part `name` with tool rotation R and
    part-frame centroid offset c."""
    P=ppos(obs,name); y=yaw_of(obs,name)
    Rp=rotz(y)
    return P+Rp@np.asarray(c)+R@CUB_D

def grasp_part(env,obs,name,R,base,c=np.zeros(3),h=0.10):
    t=target_for(obs,name,R,c)
    obs,blk,m=kgoto(env,obs,t+np.array([0,0,h]),R,base)
    if blk: return obs,False,'pre:'+m
    obs,blk,m=kgoto(env,obs,t,R,base)
    if blk: return obs,False,'app:'+m
    obs=grip(env,obs,-1.0,n=1)
    return obs,rfeat(obs,'grasp_active')>0.5,m

def place_part(env,obs,name,T,R,base,h=0.10):
    """Move so that part `name`'s reported pose reaches world T (rotation R)."""
    c=ppos(obs,name)-kpos(obs,base)          # current part offset from linpos
    obs,blk,m=kgoto(env,obs,np.array(T)+np.array([0,0,h])-c,R,base)
    if blk: return obs,'pre:'+m
    obs,blk,m=kgoto(env,obs,np.array(T)-c,R,base)
    return obs,m

def lift_to(env,obs,name,z,R,base):
    p=ppos(obs,name); return place_part(env,obs,name,[p[0],p[1],z],R,base,h=0.0)

def carry(env,obs,name,T,R,base,hi=0.32):
    """staged: lift straight up, translate at height, then descend."""
    p=ppos(obs,name)
    obs,m=place_part(env,obs,name,[p[0],p[1],hi],R,base,h=0.0)
    if m!='ok': return obs,'lift:'+m
    obs,m=place_part(env,obs,name,[T[0],T[1],hi],R,base,h=0.0)
    if m!='ok': return obs,'move:'+m
    obs,m=place_part(env,obs,name,[T[0],T[1],T[2]],R,base,h=0.0)
    return obs,('desc:'+m if m!='ok' else 'ok')

def cmove(env,obs,name,T,R,base,step=0.02):
    """Cartesian-interpolated move of part `name` (or of linpos if name=None) to T."""
    T=np.array(T,float)
    for it in range(200):
        if name is None:
            cur=kpos(obs,base); c=np.zeros(3)
        else:
            cur=ppos(obs,name); c=cur-kpos(obs,base)
        d=T-cur; n=np.linalg.norm(d)
        if n<2e-3: return obs,'ok'
        w=cur+d*min(1.0,step/n)
        obs,blk,m=kgoto(env,obs,w-c,R,base,maxsteps=12)
        if blk: return obs,'blocked@%s'%np.round(cur,3)
    return obs,'slow'
