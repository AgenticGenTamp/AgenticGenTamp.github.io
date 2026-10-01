from validate import run
from concurrent.futures import ThreadPoolExecutor
jobs=[(seed,count) for seed in [17,23,71,123] for count in [1,2,3,4,6,8]]
with ThreadPoolExecutor(4) as pool:
 results=list(pool.map(lambda sc:run(*sc),jobs))
print('TOTAL',sum(results),len(results),flush=True)
