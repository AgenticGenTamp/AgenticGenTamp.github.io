import sys,json
for f in sys.argv[1:]:
    for l in open(f):
        if l[:2] in ('A ','B ','C ','D ','E ','F ','G ','H ','K ','L ','M '):
            d=json.loads(l[2:]); print(l[0], {k:d[k] for k in ('open_cmd','dz','dy','dx','yaw_off','nclose') if k in d}, d['ok'], 'push',d['push'],'rise',d['rise'],'cmove',d['cmove'],'held',d['held'])
        elif 'acq' in l or 'Error' in l: print(f, l.strip()[:150])
