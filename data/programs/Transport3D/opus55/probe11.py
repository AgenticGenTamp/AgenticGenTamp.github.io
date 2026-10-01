from util import *
e=E(1)
c=e.o('cube0')
e.base_to(c['pose_x']-0.6, c['pose_y'])
e.ee_to([0.5,0,-0.1]); e.ee_to([0.5,0,-0.2])
def pr(t):
    r=e.rob(); print(t,'fs',round(r['finger_state'],3),'ga',r['grasp_active'],[round(r[k],4) for k in RF[12:15]], {k:round(v,4) for k,v in e.o('cube0').items() if k in ('pose_x','pose_y','pose_z','grasp_active')})
e.gripper(-1); pr('close')
print(e.ee_to([0.5,0,-0.1])); pr('lift')
for z in [-0.15,-0.19,-0.2,-0.205]:
    print(e.ee_to([0.5,0,z])); pr('lower %f'%z)
    e.gripper(1); pr('open')
e.ee_to([0.5,0,-0.1]); pr('lift')
