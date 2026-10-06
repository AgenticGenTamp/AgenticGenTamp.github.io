import sys, subprocess
from concurrent.futures import ThreadPoolExecutor
n=sys.argv[1]; seeds=sys.argv[2].split(','); w=int(sys.argv[3]) if len(sys.argv)>3 else 4
def go(s):
    out=subprocess.run(['/robocode/.venv/bin/python','run_test.py',n,s],capture_output=True,text=True).stdout
    return [l for l in out.splitlines() if l.startswith('seed')]
with ThreadPoolExecutor(w) as ex:
    for r in ex.map(go,seeds): print(*r, flush=True)
