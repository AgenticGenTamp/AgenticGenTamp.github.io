import numpy as np, kin
np.set_printoptions(precision=4,suppress=True)
ax,ay,br=-0.15+0.12*np.cos(3.13),-0.24+0.12*np.sin(3.13),3.13
tgt=np.array([-0.6009,-0.2319,0.025]); MZ=0.269
kw=dict(base_x=ax,base_y=ay,base_rot=br,mount=(0,0,MZ),return_info=True)
# seed = solution at z=0.4 (what a waypoint approach leaves you in)
q4,i4=kin.ik_top_down(np.array([tgt[0],tgt[1],0.40]),yaw=br,q_init=kin.Q_HOME,**kw)
print("z0.40 from HOME:",i4["success"],round(i4["pos_err"],5),"q",np.round(q4,3))
for name,qi in [("Q_HOME",kin.Q_HOME),("Q_RETRACT",kin.Q_RETRACT),("q@z0.4",q4)]:
    q,i=kin.ik_top_down(tgt,yaw=br,q_init=qi,**kw)
    T=kin.fk(q,base_x=ax,base_y=ay,base_rot=br,mount=(0,0,MZ))
    print("%-10s succ=%s pos_err=%.5f fk=%s j7=%.3f"%(name,i["success"],i["pos_err"],np.round(T[:3,3],4),q[6]))
# does raising max_iters/restarts fix the bad seed?
for mi,rs,tb in [(60,4,0.08),(200,8,0.5),(400,16,2.0)]:
    q,i=kin.ik_top_down(tgt,yaw=br,q_init=q4,max_iters=mi,restarts=rs,time_budget=tb,**kw)
    print("q@z0.4 iters=%d rst=%d succ=%s err=%.5f"%(mi,rs,i["success"],i["pos_err"]))
