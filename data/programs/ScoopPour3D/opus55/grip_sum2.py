import sys,json
for f in sys.argv[1:]:
    for l in open(f):
        if l[1]==' ' and l[2]=='{':
            d=json.loads(l[2:]); k={k:d[k] for k in ('dz','dy','dx','yaw_off','nclose') if k in d}
            if 'yaw_off' in k: k['yaw_off']=round(k['yaw_off']*57.3)
            print(l[0],d['open_cmd'],k,'OK' if d['ok'] else 'fail','terr',[round(v*1000,1) for v in d['tool_err']],'push',round(d['push']*1000,1),'cmove',[round(v*1000,1) for v in d['cmove']],'held',[round(v*1000,1) for v in d['held']])
        elif 'acq' in l: print(f,l.strip()[:80])
