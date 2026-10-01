from lib import *
e=E(0); e.grasp_hook(1.2)
print(e.obs[:3],e.obs[9:12])
# move left until stuck
for i in range(40):
    p=e.obs.copy(); e.st([-0.01,0,0,0,1])
    if np.allclose(p[:2],e.obs[:2]): print('stuck left',e.obs[:3],e.obs[9:12]); break
# rotate hook so long arm points up: hook theta pi/2 ; move up
x,y,th=e.robot_for_hook(1.0,2.0,np.pi/2)
print(e.goto(e.obs[0],e.obs[1],th,vac=1))
print(e.goto(1.0,e.obs[1],th,vac=1),e.obs[:3],e.obs[9:12])
for i in range(80):
    p=e.obs.copy(); e.st([0,0.01,0,0,1])
    if np.allclose(p[:2],e.obs[:2]): print('stuck up',e.obs[:3],e.obs[9:12]); break
print('final',e.obs[:3],e.obs[9:12])
