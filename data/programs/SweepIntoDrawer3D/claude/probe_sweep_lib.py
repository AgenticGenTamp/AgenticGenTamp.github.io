import numpy as np
from ctrl2 import moveto, move_base, RDOWN, tip_to_fk, JLO, JHI
from ik import ik

def w2l(obs, wx, wy):
    bx,by,yaw = obs[125],obs[126],obs[127]
    dx=wx-bx; dy=wy-by
    lx=np.cos(yaw)*dx+np.sin(yaw)*dy
    ly=-np.sin(yaw)*dx+np.cos(yaw)*dy
    return lx,ly

def l2w(obs, lx, ly):
    bx,by,yaw = obs[125],obs[126],obs[127]
    wx = bx + np.cos(yaw)*lx - np.sin(yaw)*ly
    wy = by + np.sin(yaw)*lx + np.cos(yaw)*ly
    return wx,wy

def tipw(obs, wx, wy, wz):
    lx,ly=w2l(obs,wx,wy)
    return [lx,ly,wz]

def cubes(obs):
    return np.asarray(obs[0:80]).reshape(5,16)[:,0:3]

def wiper(obs):
    return np.asarray(obs[147:151])[0:3]

def movew(env, obs, wp, steps=120, grip=0.0, **kw):
    return moveto(env, obs, tipw(obs,*wp), RDOWN, steps=steps, grip=grip, **kw)

def ikres_world(obs, wx, wy, wz, R=RDOWN):
    p=tip_to_fk(tipw(obs,wx,wy,wz),R)
    q,res=ik(np.array(p),R,obs[128:135].copy())
    clip=np.abs(np.clip(q,JLO,JHI)-q).max()
    return res, clip
