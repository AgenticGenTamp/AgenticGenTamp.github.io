"""Planar downward grasp geometry for the seven-joint arm."""
import numpy as np

def grasp_joints(z, reach=.65):
    upper=.42076
    forearm=.31436
    vertical=z+.042645
    c=(reach*reach+vertical*vertical-upper*upper-forearm*forearm)/(2*upper*forearm)
    elbow=-np.arccos(np.clip(c,-1.,1.))
    shoulder=np.arctan2(reach,vertical)-np.arctan2(forearm*np.sin(-elbow),upper+forearm*np.cos(elbow))
    wrist=shoulder-elbow-np.pi
    return np.array([0.,shoulder,np.pi,elbow,0.,wrist,np.pi/2])
