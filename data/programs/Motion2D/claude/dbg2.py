from env_client import make_env
from approach import GeneratedApproach
env=make_env(); ap=GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=10); ap.reset(obs,info)
print("gdir",ap.gdir,"r",ap.radius,"gf",ap.gf,"gh",ap.gh)
a=(1.9237,2.0164); b=(1.9737,2.041)
print("free_seg",ap._free_seg(a[0],a[1],b[0],b[1],ap.radius))
print("free_pt a",ap._free_pt(*a,ap.radius),"free_pt b",ap._free_pt(*b,ap.radius))
print("box a",ap._box_free(*a),"box b",ap._box_free(*b))
for rc in ap._rects:
    print(rc.bbox, "segd",round(rc.seg_dist(a[0],a[1],b[0],b[1]),4))
