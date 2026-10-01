from calib_util import *
import sys, json
def rot(ax,t):
    ax=np.array(ax,float); K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K
