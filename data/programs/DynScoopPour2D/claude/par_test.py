import subprocess, sys, concurrent.futures as cf
seeds=[int(x) for x in sys.argv[1:]]
def run(s):
    r=subprocess.run(['/opt/robocode-strict/bin/python','test_approach.py',str(s)],capture_output=True,text=True)
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'ERR '+r.stderr[-200:]
with cf.ThreadPoolExecutor(8) as ex:
    for line in ex.map(run, seeds): print(line, flush=True)
