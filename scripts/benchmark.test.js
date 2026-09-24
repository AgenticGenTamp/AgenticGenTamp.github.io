import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {summarize,selectEnvironments} from '../benchmark.js';
const data=JSON.parse(readFileSync(new URL('../data/benchmark.json',import.meta.url)));
test('full coverage ranks exclude a partial-coverage planner and source reference',()=>{
 const rows=summarize(data);const planner=rows.find(x=>x.id==='planner');
 assert.equal(planner.count,16);assert.equal(planner.rank,null);
 assert.equal(rows.find(x=>x.id==='source').rank,null);
 assert.equal(rows.find(x=>x.id==='astra').rank,1);
 assert.equal(rows.find(x=>x.id==='claude').rank,2);
 assert.ok(Math.abs(planner.mean-.466875)<1e-8);
});
test('matched planner comparison uses precisely the same 16 environments for every method',()=>{
 const rows=summarize(data,'shared');assert.equal(selectEnvironments(data,'shared').length,16);
 for(const r of rows){assert.equal(r.count,16);assert.equal(r.total,16);}
 assert.equal(rows.find(x=>x.id==='planner').rank,4);
});
test('zero success is retained; unavailable planner entries are never turned into zeros',()=>{
 const family=selectEnvironments(data,'all','Dynamic3D');assert.equal(family.length,10);
 const planner=summarize(data,'all','Dynamic3D').find(x=>x.id==='planner');
 assert.equal(planner.count,3);assert.ok(Math.abs(planner.mean-1.04/3)<1e-10);
 assert.equal(data.environments.find(x=>x.id==='SweepIntoDrawer3D').results.planner.mean,0);
 assert.equal(data.environments.find(x=>x.id==='ScoopPour3D').results.planner,null);
});
test('all filter combinations preserve counts and score sort direction',()=>{
 for(const scope of ['all','shared'])for(const family of ['all',...new Set(data.environments.map(x=>x.family))]){
   const rows=summarize(data,scope,family,false).filter(x=>x.kind!=='reference'&&x.mean!==null);
   for(let i=1;i<rows.length;i++)assert.ok(rows[i].mean>=rows[i-1].mean);
   for(const row of rows)assert.ok(row.count<=row.total);
 }
});
test('printed run ranges contain every mean',()=>{
 for(const e of data.environments)for(const r of Object.values(e.results))if(r){assert.ok(r.min<=r.mean&&r.mean<=r.max);assert.ok(r.min>=0&&r.max<=1);}
});
test('both + source rows stay unranked references with complete coverage',()=>{
 const rows=summarize(data);const refs=rows.filter(x=>x.kind==='reference');
 assert.deepEqual(refs.map(x=>x.id),['source','astraSource']);
 for(const r of refs){assert.equal(r.rank,null);assert.equal(r.count,28);}
 assert.deepEqual(rows.slice(-2).map(x=>x.id),['source','astraSource']);
});
