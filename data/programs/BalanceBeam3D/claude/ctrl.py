import os
import numpy as np
import kinova
MOUNT=np.array([float(os.environ.get("MX","0.13")),float(os.environ.get("MY","0.0")),0.45])
TOOL=float(os.environ.get("TOOL","0.20"))
ARM_GAIN=0.25; BASE_GAIN=0.87; YAW_GAIN=0.994
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)

def Rz(t):
    c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def w2a(p, base):
    return Rz(base[2]).T@(p-np.array([base[0],base[1],0.0]))-MOUNT

def ee_world(obs, tool=TOOL):
    b=obs[16:19]
    return np.array([b[0],b[1],0.0])+Rz(b[2])@(MOUNT+kinova.fk(obs[19:26],tool)[:3,3])

def grip_R(yaw):
    return Rz(yaw)@Rdown
