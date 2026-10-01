import numpy as np, ctrl
from probe_base_lib import E
e=E(0); s=e.obs
names=s.get_object_names()
print(names)
for n in names:
    o=s.get_object_from_name(n)
    try: fs=list(s.get_field_names(o))
    except Exception:
        fs=None
    print(n, fs if fs and len(fs)<15 else (fs[:12] if fs else None))
e.close()
