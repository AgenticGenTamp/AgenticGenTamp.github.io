import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {summarize,selectEnvironments,experimentTotals} from '../benchmark.js';
const data=JSON.parse(readFileSync(new URL('../data/benchmark.json',import.meta.url)));
test('full coverage ranks exclude a partial-coverage planner and source reference',()=>{
 const rows=summarize(data);const planner=rows.find(x=>x.id==='planner');
 assert.equal(planner.count,16);assert.equal(planner.rank,null);
 assert.equal(rows.find(x=>x.id==='source').rank,null);
 assert.equal(rows.find(x=>x.id==='astra').rank,2);
 assert.equal(rows.find(x=>x.id==='claude').rank,3);
 assert.ok(Math.abs(planner.mean-.466875)<1e-8);
});
test('planner subset compares complete results on the same 16 environments',()=>{
 const rows=summarize(data,'shared');assert.equal(selectEnvironments(data,'shared').length,16);
 for(const r of rows){assert.equal(r.count,16);assert.equal(r.total,16);} assert.equal(rows.find(x=>x.id==='opus55').rank,1);
 assert.equal(rows.find(x=>x.id==='planner').rank,5);
});
test('zero success is retained; unavailable planner entries are never turned into zeros',()=>{
 const family=selectEnvironments(data,'all','Dynamic3D');assert.equal(family.length,10);
 const planner=summarize(data,'all','Dynamic3D').find(x=>x.id==='planner');
 assert.equal(planner.count,3);assert.ok(Math.abs(planner.mean-1.04/3)<1e-10);
 assert.equal(data.environments.find(x=>x.id==='SweepIntoDrawer3D').results.planner.mean,0);
 assert.equal(data.environments.find(x=>x.id==='ScoopPour3D').results.planner,null);
});
test('all filter combinations preserve counts and score sort direction',()=>{
 for(const scope of ['all','shared'])for(const family of ['all',...new Set(data.environments.map(x=>x.family))])for(const descending of [true,false]){
   const summary=summarize(data,scope,family,descending);
   for(const reference of [false,true]){
     const rows=summary.filter(x=>(x.kind==='reference')===reference&&x.mean!==null);
     for(let i=1;i<rows.length;i++)assert.ok(descending?rows[i].mean<=rows[i-1].mean:rows[i].mean>=rows[i-1].mean);
     for(const row of rows)assert.ok(row.count<=row.total);
   }
   const firstReference=summary.findIndex(x=>x.kind==='reference');
   assert.ok(summary.slice(firstReference).every(x=>x.kind==='reference'));
 }
});
test('printed run ranges contain every mean',()=>{
 for(const e of data.environments)for(const r of Object.values(e.results))if(r){assert.ok(r.min<=r.mean&&r.mean<=r.max);assert.ok(r.min>=0&&r.max<=1);}
});
test('both + source rows stay unranked references with complete coverage',()=>{
 const rows=summarize(data);const refs=rows.filter(x=>x.kind==='reference');
 assert.deepEqual(refs.map(x=>x.id),['astraSource','source']);
 for(const r of refs){assert.equal(r.rank,null);assert.equal(r.count,28);}
 assert.deepEqual(rows.slice(-2).map(x=>x.id),['astraSource','source']);
 assert.deepEqual(summarize(data,'all','all',false).slice(-2).map(x=>x.id),['source','astraSource']);
});

test('Opus and Astra use all 28 environments with complete five-run coverage',()=>{
 const rows=summarize(data);assert.equal(selectEnvironments(data).length,28);
 for(const id of ['opus55','astra']){const row=rows.find(r=>r.id===id);assert.equal(row.count,28);assert.equal(row.total,28);}
 const opus=rows.find(r=>r.id==='opus55');assert.equal(opus.rank,1);
 assert.ok(Math.abs(opus.mean-.9471428571428572)<1e-10);
 assert.ok(Math.abs(rows.find(r=>r.id==='astra').mean-.8567857142857143)<1e-10);
});
test('Rovers has a measured five-run range and no remaining missing seeds',()=>{
 assert.deepEqual(data.environments.find(e=>e.id==='PddlRovers').results.opus55,{mean:1,min:1,max:1});
 assert.deepEqual(data.coverage.opus55.pending,[]);
 assert.equal(data.coverage.opus55.completeRuns,140);
 assert.equal(data.coverage.opus55.completeEnvironments,28);
 assert.ok(data.environments.every(e=>e.results.opus55 && !e.results.opus55.provisional));
});

const runSources=Object.fromEntries(data.methods.filter(m=>m.resultSource).map(m=>[m.id,JSON.parse(readFileSync(new URL('../'+m.resultSource,import.meta.url)))]));
test('experiment totals count paper programs plus audited runs, excluding planners',()=>{
 assert.deepEqual(experimentTotals(data,runSources),{environments:28,programs:1120,episodes:112000,additional:[
  {backend:'Codex with GPT-6 Astra (high)',programs:280},
  {backend:'Claude Code with Opus 5.5 (high)',programs:140}
 ]});
});
test('a newly registered model contributes completed runs even before an environment mean is ready',()=>{
 const next=structuredClone(data);
 next.methods.push({id:'newModel',backend:'New model',resultSource:'data/new-results.json'});
 next.environments.forEach(e=>e.results.newModel=null);
 const sources={...runSources,newModel:{method:'newModel',environments:[
  {id:next.environments[0].id,runs:[{episodes:100},{episodes:80}]},
  {id:'DroppedEnvironment',runs:[{episodes:100}]}
 ]}};
 const totals=experimentTotals(next,sources);
 assert.equal(totals.programs,1122);assert.equal(totals.episodes,112180);
 assert.deepEqual(totals.additional.at(-1),{backend:'New model',programs:2});
});
test('removing an environment removes its programs from both paper and run-based totals',()=>{
 const subset={...data,environments:data.environments.filter(e=>e.id!=='PddlRovers')};
 assert.equal(experimentTotals(subset,runSources).programs,1080);
 assert.equal(experimentTotals(subset,runSources).episodes,108000);
});
test('missing run sources never silently produce a lower total',()=>{
 assert.throws(()=>experimentTotals(data,{}),/Missing or mismatched run source/);
});
