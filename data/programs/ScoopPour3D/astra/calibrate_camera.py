import numpy as np
pts=[]
for y,uvback,uvfront,uvbottom in [(-.350876,(183,287),(135,395),(136,424)),(-.050876,(299,287),(294,395),(294,424)),(.050384,(341,287),(346,395),(346,424)),(.350384,(457,287),(505,395),(505,424))]:
 for xyz,uv in [((.275,y,.528),uvback),((.725,y,.528),uvfront),((.725,y,.458),uvbottom)]:pts.append((xyz,uv))
a=[]
for xyz,(u,v) in pts:
 X=np.r_[xyz,1];a.extend([np.r_[X,np.zeros(4),-u*X],np.r_[np.zeros(4),X,-v*X]])
_,_,V=np.linalg.svd(a);P=V[-1].reshape(3,4)
def proj(xyz):
 p=P@np.r_[xyz,1];return p[:2]/p[2]
print('error',np.mean([np.linalg.norm(proj(x)-uv) for x,uv in pts]))
print('cube0',proj([.52968,-.17591,.48251]))
for p in [[.517,-.180,.646],[.532,-.169,.505],[-.07911,.04934,.60866]]:
 print('pred',p,proj(p))
 for dz in [0,-.04,-.08,-.12]: print('dz',dz,proj(np.array(p)+[0,0,dz]))
print('P',P)
from scipy.optimize import least_squares
pred=np.array([[-.07911,.04934,.60866],[.517,-.180,.646],[.532,-.169,.505]])
uv=np.array([[338,228],[228,339],[237,390]])
fit=least_squares(lambda off: np.ravel([proj(p+off)-u for p,u in zip(pred,uv)]),[.1,0,-.04])
print('offset',fit.x,'residual',fit.fun)
for p in pred:print('corrected',proj(p+fit.x))
