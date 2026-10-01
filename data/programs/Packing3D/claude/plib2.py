import numpy as np
import kin
from probe_lib import *
from lib_probe import getq, move_to, Rdown

def kgoto(env,obs,p_world,R,base,maxsteps=60):
    """Drive so that kin.linpos == p_world (world coords) with rotation R."""
    tgt=np.array([p_world[0]-base[0],p_world[1]-base[1],p_world[2]])
    q0=getq(obs)
    qd,p,Rr=kin.ikin(q0,tgt,R)
    if np.linalg.norm(p-tgt)>1e-3 or np.linalg.norm(Rr-R)>1e-2:
        return obs,True,'ik_fail'
    obs,blk=move_to(env,obs,qd,maxsteps=maxsteps)
    err=np.linalg.norm(kin.linpos(getq(obs))-tgt)
    return obs,(blk or err>3e-3),('blocked' if blk else ('err%.4f'%err if err>3e-3 else 'ok'))

def kpos(obs,base):
    return kin.linpos(getq(obs))+np.array([base[0],base[1],0.0])

def try_grasp(env,obs,part,delta_g,R,base,approach_h=0.10):
    """delta_g: offset in gripper frame added to the part position."""
    P=ppos(obs,part)
    tgt=P+R@np.array(delta_g)
    obs,blk,m=kgoto(env,obs,tgt+np.array([0,0,approach_h]),R,base)
    if blk: return obs,False,'pre:'+m
    obs,blk,m=kgoto(env,obs,tgt,R,base)
    if blk: return obs,False,'app:'+m
    obs=grip(env,obs,-1.0,n=1)
    return obs,rfeat(obs,'grasp_active')>0.5,m
