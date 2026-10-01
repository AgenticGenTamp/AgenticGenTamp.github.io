import numpy as np, kin
tgt=np.array([-0.6009,-0.2319,0.025]); FWD=0.12; MZ=0.269
print("d_horiz : poserr (yaw=base_rot)")
for d in [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]:
    # base placed at distance d from cube along +x direction, facing cube (rot=pi)
    bx,by,br=tgt[0]+d,tgt[1],np.pi
    ax,ay=bx+FWD*np.cos(br),by+FWD*np.sin(br)
    out=[]
    for zt in [0.025,0.05,0.10]:
        q,info=kin.ik(np.array([tgt[0],tgt[1],zt]),target_R=kin.top_down_quat and None,q_init=kin.Q_HOME,
                      base_x=ax,base_y=ay,base_rot=br,mount=(0,0,MZ),return_info=True) if False else (None,None)
        q,info=kin.ik_top_down(np.array([tgt[0],tgt[1],zt]),yaw=br,q_init=kin.Q_HOME,base_x=ax,base_y=ay,
                               base_rot=br,mount=(0,0,MZ),return_info=True)
        out.append("z%.3f:%s %.4f"%(zt,"OK" if info["success"] else "NO",info["pos_err"]))
    print("d=%.2f (armdist %.3f)"%(d,d-FWD)," | ".join(out))
