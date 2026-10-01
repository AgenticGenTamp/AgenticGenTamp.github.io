from probe_vis_lib import *
env=make_env()
for seed,pil,blocked in [(20,'obstacle7',False),(22,'obstacle10',False),(23,'obstacle5',True),(39,'obstacle9',True),(39,'obstacle5',True),(48,'obstacle10',True),(60,'obstacle6',True)]:
    obs,info=env.reset(seed=seed, options={'object_count':1})
    f=feats(obs,pil); o=feats(obs,'objective0')
    print("seed%-3d %-11s blocked=%-5s  pillar z=%.4f half_z=%.4f hx=%.4f  obj z=%.4f"%(seed,pil,blocked,f['z'],f['half_z'],f['half_x'],o['z']))
env.close()
