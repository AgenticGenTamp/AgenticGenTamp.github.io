import sys
from test_approach import run
from concurrent.futures import ThreadPoolExecutor, as_completed
seeds=range(int(sys.argv[1]),int(sys.argv[2]))
with ThreadPoolExecutor(16) as ex:
    futs={ex.submit(run,s):s for s in seeds}
    for f in as_completed(futs):
        ok,n=f.result(); print(futs[f],ok,n,flush=True)
