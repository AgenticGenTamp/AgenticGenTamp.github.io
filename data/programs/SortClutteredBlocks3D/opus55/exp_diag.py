import sys, json, numpy as np
import exp_approach as A
from env_client import make_env
seed=int(sys.argv[1]); cfg=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
for k,v in cfg.items(): setattr(A,k,v)
if "G_OPEN" in cfg: A.OPEN_GAP=0.085*(1-A.G_OPEN)
env=make_env(); obs,info=env.reset(seed=seed,options={"object_count":20})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
gp=lambda o: float(o.get(o.get_object_from_name("robot"),"pos_gripper"))
last=None; cur=None; out=[]
for t in range(700):
    a=ap.get_action(obs); ph=ap.phase
    if ph=="close" and last!="close":
        c=ap.cubes[ap.target]; bw=ap._bracelet_world()
        y,tl=A.cube_face_yaw(c["quat"])
        R=A.fk_arm(ap.q,0.0)[1]; gy=ap.base[2]+np.arctan2(R[1,0],R[0,0])
        dy=np.degrees((gy-y+np.pi/4)%(np.pi/2)-np.pi/4); dphi=np.degrees((gy-ap.phi+np.pi/2)%np.pi-np.pi/2)
        u=np.array([np.cos(ap.phi),np.sin(ap.phi)]); v=np.array([-u[1],u[0]])
        nb=sorted([(np.linalg.norm(o["p"][:2]-c["p"][:2]),abs((o["p"][:2]-c["p"][:2])@u)*1000,abs((o["p"][:2]-c["p"][:2])@v)*1000,(o["p"][2]-c["p"][2])*1000,m) for m,o in ap.cubes.items() if m!=ap.target])[:3]
        nbs=" ".join(f"({a:.0f},{b:.0f},{d:+.0f}{'s' if ap._cube_color(m)==ap._cube_color(ap.target) else ''})" for _,a,b,d,m in nb)
        cur=dict(nbs=nbs,clr=ap._axis_clearance(ap.target,ap.phi),xy=np.linalg.norm(bw[:2]-c["p"][:2])*1000,
                 dz=(bw[2]-c["p"][2]-A.GRASP_DZ)*1000,g0=gp(obs),cz=c["p"][2],gtr=[],dy=dy,dphi=dphi,tl=np.degrees(tl))
    if cur is not None and ph in("close","lift"): cur["gtr"].append(round(gp(obs),2)); cur["zmax"]=max(cur.get("zmax",0),ap.cubes[ap.target]["p"][2]-cur["cz"])
    if last=="lift" and ph!="lift" and cur:
        print(f"{ph[:5]:5s} clr={cur['clr']*1000:+5.1f} xy={cur['xy']:.1f}mm dz={cur['dz']:+.1f}mm cube_dz={cur['zmax']*1000:5.1f}mm tilt={cur['tl']:.0f} cz={cur['cz']*1000:.0f} nb(al,pe,dz)={cur['nbs']}")
        cur=None
    last=ph; obs,r,term,trunc,info=env.step(a)
    if term: break
