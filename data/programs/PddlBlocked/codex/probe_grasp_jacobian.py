"""Use a rigidly held blocker as a proxy for the otherwise-hidden tool pose."""

import math
import numpy as np

from env_client import make_env


def get(s, name, feature):
    return s.get(s.get_object_from_name(name), feature)


def block_pose(s):
    p = np.array([get(s, "blocker", f) for f in ("pose_x", "pose_y", "pose_z")])
    q = np.array([get(s, "blocker", f) for f in ("pose_qx", "pose_qy", "pose_qz", "pose_qw")])
    return p, q


def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw,
                     aw*bw-ax*bx-ay*by-az*bz])


def rotvec(qnew, qold):
    dq = qmul(qnew, np.array([-qold[0], -qold[1], -qold[2], qold[3]]))
    if dq[3] < 0: dq = -dq
    dq /= np.linalg.norm(dq)
    n = np.linalg.norm(dq[:3])
    return dq[:3] / max(n, 1e-12) * (2 * math.atan2(n, dq[3]))


def main():
    env = make_env()
    try:
        s, _ = env.reset(seed=0)
        # Reproduce search result: lift shoulder and move into the blocker.
        tx, ty, lift = 3.6888845, -0.2602724, 0.45
        while abs(get(s,"robot","base_x")-tx)>1e-5 or abs(get(s,"robot","base_y")-ty)>1e-5 or lift > 1e-5:
            a=np.zeros(11,np.float32)
            a[0]=np.clip(tx-get(s,"robot","base_x"),-.2,.2)
            a[1]=np.clip(ty-get(s,"robot","base_y"),-.2,.2)
            a[4]=min(.2,lift); lift-=a[4]
            s,*_=env.step(a)
        a=np.zeros(11,np.float32);a[10]=-1;s,*_=env.step(a)
        print("grasp",get(s,"robot","grasp_active"),"base/joints",[round(get(s,"robot",f),6) for f in
              ("base_x","base_y","base_rot","joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7")])
        print("grasp_tf",[round(get(s,"robot",f),6) for f in
              ("grasp_tf_x","grasp_tf_y","grasp_tf_z","grasp_tf_qx","grasp_tf_qy","grasp_tf_qz","grasp_tf_qw")])
        # Lift/pull the block clear of the pen before measuring unconstrained axes.
        a=np.zeros(11,np.float32);a[4]=-.2;s,*_=env.step(a)
        print("after clearance",[round(x,6) for x in block_pose(s)[0]],
              "joint_2",round(get(s,"robot","joint_2"),6))
        for i in range(10):
            feature=("base_x","base_y","base_rot","joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7")[i]
            for sign in (1.,-1.):
                p0,q0=block_pose(s); oldq=get(s,"robot",feature)
                a=np.zeros(11,np.float32);a[i]=sign*.01;s2,*_=env.step(a)
                p1,q1=block_pose(s2); actual=get(s2,"robot",feature)-oldq
                accepted=abs(actual)>.005
                print(i,"sign",int(sign),"accepted",accepted,"dp/dq",np.round((p1-p0)/(sign*.01),6),
                      "drot/dq",np.round(rotvec(q1,q0)/(sign*.01),6))
                if accepted:
                    a[i]=-sign*.01;s,*_=env.step(a)
                    break
                s=s2
    finally:
        env.close()


if __name__ == "__main__":
    main()
