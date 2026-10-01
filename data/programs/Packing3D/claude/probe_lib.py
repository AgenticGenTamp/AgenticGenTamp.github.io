import numpy as np
from env_client import make_env
from fk import fk
from ik import ik
from lib_probe import getq, move_to, Rdown, JNAMES

def rotz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def rotx(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def roty(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rb(obs):
    r=obs.get_object_from_name('robot')
    return np.array([obs.get(r,'pos_base_x'),obs.get(r,'pos_base_y')])

def rfeat(obs,f):
    return obs.get(obs.get_object_from_name('robot'),f)

def pfeat(obs,name,f):
    return obs.get(obs.get_object_from_name(name),f)

def ppos(obs,name):
    o=obs.get_object_from_name(name)
    return np.array([obs.get(o,'pose_x'),obs.get(o,'pose_y'),obs.get(o,'pose_z')])

def goto(env,obs,xyz,R,base,maxsteps=60,close=0.0):
    q=getq(obs)
    tgt=np.array([xyz[0]-base[0],xyz[1]-base[1],xyz[2]])
    qd,M=ik(q,tgt,R)
    if np.linalg.norm(M[:3,3]-tgt)>2e-3:
        return obs, True, 'ik_fail'
    obs,blk=move_to(env,obs,qd,maxsteps=maxsteps,close=close)
    err=np.linalg.norm(fk(getq(obs))[:3,3]-tgt)
    return obs, (blk or err>5e-3), ('blocked' if blk else ('err%.3f'%err if err>5e-3 else 'ok'))

def grip(env,obs,val,n=2):
    a=np.zeros(11); a[10]=val
    for _ in range(n):
        obs,r,t,tr,i=env.step(a)
    return obs

def fkpos(obs):
    return fk(getq(obs))[:3,3]
