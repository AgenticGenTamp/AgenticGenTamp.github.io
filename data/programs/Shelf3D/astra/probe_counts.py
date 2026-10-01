from env_client import make_env
for n in [0,1,3,5,10]:
    e=make_env()
    try:
        s,i=e.reset(seed=1,options={'object_count':n})
        print('COUNT',n,'INFO',i,flush=True)
        for name in sorted(s.get_object_names()):
            o=s.get_object_from_name(name)
            if name!='robot': print(name,[round(float(s.get(o,f)),4) for f in ['x','y','z']],flush=True)
    except Exception as ex: print(type(ex).__name__,str(ex),flush=True)
    e.close()
