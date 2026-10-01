from lib import *
e=E(0)
e.grasp_hook(1.2)
print('grasped',e.obs[:12],e.off,e.dth)
bx,by=e.obs[20:22]
print('button',bx,by,'target',e.obs[29:31])
th=np.pi/2
vx=bx-0.05-0.03-0.025; vy=by+0.15
print(e.robot_for_hook(vx,vy,th))
# first rotate in place
x,y,rth=e.robot_for_hook(e.obs[9],e.obs[10],th)
print(e.goto(e.obs[0],e.obs[1],rth,vac=1))
print('rot', e.obs[:3], e.obs[9:12])
x,y,rth=e.robot_for_hook(vx,vy,th)
print(e.goto(x,e.obs[1],rth,vac=1)); print(e.obs[:3],e.obs[9:12])
print(e.goto(x,y,rth,vac=1)); print(e.obs[:3],e.obs[9:12])
for i in range(12):
    e.st([0.01,0,0,0,1]); print(e.obs[0:2],e.obs[9:11],'btn',e.obs[20:23],e.obs[24:27],e.obs[33:36])
