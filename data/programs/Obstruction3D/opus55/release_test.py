from th import *
h=H(1); p=h.pose('target_block'); he=h.he('target_block'); top=p[2]+he[2]
h.goto([p[0],p[1],top+0.1],0)
for dz in [-0.01,-0.02,-0.03]:
    h.goto([p[0],p[1],top+dz],0); h.grip(-1)
    if h.grasped(): break
print('grasped at dz',dz,h.grasped(), h.obs.get(h.R,'grasp_tf_z'))
h.goto([p[0],p[1],top+0.1],0); print('lifted block', h.pose('target_block')[:3].round(4))
h.goto([p[0]+0.05,p[1]-0.1,top+0.1],0)
b1=h.pose('target_block'); h.grip(1); b2=h.pose('target_block')
print('release in air: before',b1[:3].round(4),'after',b2[:3].round(4),'grasp',h.grasped())
for i in range(3): h.step(np.zeros(11))
print('after 3 steps', h.pose('target_block')[:3].round(4))
for i in range(3):
    h.grip(1); print('open again', h.grasped())
h.goto([p[0]+0.05,p[1]-0.1,top+0.0],0); print('lowered', h.pose('target_block')[:3].round(4))
h.grip(1); print('open near table', h.grasped(), h.pose('target_block')[:3].round(4))
h.goto([p[0]+0.05,p[1]-0.1,top-0.01],0); print('lowered more', h.pose('target_block')[:3].round(4))
h.grip(1); print('open near table', h.grasped(), h.pose('target_block')[:3].round(4))
