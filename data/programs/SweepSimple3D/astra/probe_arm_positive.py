exec(open('probe_arm_grid.py').read().split('with ThreadPoolExecutor')[0])
with ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(trial,itertools.product([1.8,2.4],[-2.55,-1.5,-.5])))
