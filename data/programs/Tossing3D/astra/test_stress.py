from test_batch import *
import json
seeds=[s for d,s in json.load(open('close_seeds.json'))][:12]
with ThreadPoolExecutor(max_workers=3) as pool:
 for r in pool.map(run,[(s,2) for s in seeds]):print(json.dumps(r),flush=True)
