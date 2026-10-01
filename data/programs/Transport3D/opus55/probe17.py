from util import *
e=E(1)
b=e.o('box0'); bx,by=b['pose_x'],b['pose_y']
print(place_and_reach2(e,bx-0.095,by,0.2,pre=0.3))
e.gripper(-1)
def pr(t):
    r=e.rob(); bb=e.o('box0'); print(t,'ga',r['grasp_active'],'fs',round(r['finger_state'],3),[round(bb[k],4) for k in ('pose_x','pose_y','pose_z','grasp_active')], np.round(fk_arm(e.q())[:3,3],4))
pr('grasp')
print(lin_to(e,[0.5,0,0.1])); pr('lift')
print(e.base_to(e.rob()['pos_base_x']-0.3, e.rob()['pos_base_y'])); pr('move')
print(lin_to(e,[0.5,0,-0.04])); pr('lower')
e.gripper(1); pr('open')
print(lin_to(e,[0.5,0,-0.045])); pr('lower')
e.gripper(1); pr('open')
print(lin_to(e,[0.5,0,0.1])); pr('lift')
