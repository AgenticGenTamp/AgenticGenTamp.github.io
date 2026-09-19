import {selectEnvironments, summarize} from './benchmark.js';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const percent = n => `${(n * 100).toFixed(1)}%`;
const whole = n => `${Math.round(n * 100)}%`;
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hero = $('#hero-video');
let heroWanted = !reduced.matches && !navigator.connection?.saveData;
function updateMotionButton() { $('#motion-toggle').textContent = hero.paused ? '▶ Background' : 'Ⅱ Background'; $('#motion-toggle').setAttribute('aria-label', hero.paused ? 'Play background video' : 'Pause background video'); }
hero.addEventListener('play', updateMotionButton);hero.addEventListener('pause', updateMotionButton);
$('#motion-toggle').addEventListener('click', () => { heroWanted = hero.paused; if (heroWanted) hero.play().catch(updateMotionButton); else hero.pause(); });
new IntersectionObserver(entries => { const visible = entries[0].isIntersecting; if (visible && heroWanted && !document.hidden) hero.play().catch(updateMotionButton); else hero.pause(); }, {threshold:0.05}).observe(hero);
reduced.addEventListener('change', () => { if (reduced.matches) {heroWanted=false;hero.pause();} });
document.addEventListener('visibilitychange',()=>{ if(document.hidden){hero.pause();$('#method-video').pause();$('#env-video').pause();$('#dialog-video').pause();pauseFilm();}else if(heroWanted && $('#top').getBoundingClientRect().bottom>0){hero.play().catch(()=>{});} });
let scrolled = false;
function navState(){const next=window.scrollY>100;if(next!==scrolled){$('#nav').classList.toggle('sticky',next);scrolled=next;}}
window.addEventListener('scroll',navState,{passive:true});navState();

// The supplied main video is a timeline-driven HTML film. Keep its original player.
function pauseFilm(){ $('#film-shell iframe')?.contentWindow?.postMessage({type:'agentamp-film-pause'},location.origin); }
function openFilm(time=0){
  const frame=document.createElement('iframe');frame.title='Project film: Coding Agents for Generalized Task and Motion Planning';frame.src=`film/?t=${time}&autoplay=1&captions=1`;frame.allow='autoplay; fullscreen';frame.allowFullscreen=true;
  $('#film-shell').replaceChildren(frame);$('#method-video').pause();$('#env-video').pause();
}
$('#film-play').addEventListener('click',()=>openFilm());
$$('[data-time]').forEach(b=>b.addEventListener('click',()=>{openFilm(Number(b.dataset.time));$('#film-shell').scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'center'});}));
new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)pauseFilm();},{threshold:0.01}).observe($('#film-shell'));
const steps=[
 {file:'shelf-floorprobe-r222',badge:'BEFORE CALIBRATION',title:'An imperfect model meets the simulator.',copy:'The agent constructs an arm model from prior knowledge, then tests it through interaction. Its initial geometry is inaccurate.',metric:'38.9',unit:'mm initial calibration RMSE'},
 {file:'shelf-calib-main-r222',badge:'FIT ROBOT GEOMETRY',title:'Measure. Calibrate. Try again.',copy:'Using a grasped block as a marker, the agent fits six geometry parameters to observed positions. The model becomes accurate enough to guide inverse kinematics.',metric:'1.8',unit:'mm calibrated RMSE'},
 {file:'shelf-heldout-100-r222',badge:'FROZEN PROGRAM · HELD-OUT TESTS',title:'The same code, across unseen instances.',copy:'The finished policy is frozen and evaluated on 100 unseen instances. It acts directly from observations, without calling an LLM at test time.',metric:'100',unit:'held-out instances per program'}
];
function selectStep(i){
 const s=steps[i];$$('[data-step]').forEach((b,j)=>{b.classList.toggle('active',j===i);b.setAttribute('aria-selected',String(j===i));b.tabIndex=j===i?0:-1;});$('#method-panel').setAttribute('aria-labelledby',`step-${i}`);
 const v=$('#method-video');v.pause();v.poster=`assets/posters/${s.file}.jpg`;v.src=`film/assets/clips/${s.file}.mp4`;v.load();
 $('#method-badge').textContent=s.badge;$('#method-title').textContent=s.title;$('#method-copy').textContent=s.copy;$('#method-metric').innerHTML=`${s.metric} <span>${s.unit}</span>`;
}
$$('[data-step]').forEach((b,i)=>{b.addEventListener('click',()=>selectStep(i));b.addEventListener('keydown',e=>{let next=i;if(e.key==='ArrowDown'||e.key==='ArrowRight')next=(i+1)%3;else if(e.key==='ArrowUp'||e.key==='ArrowLeft')next=(i+2)%3;else if(e.key==='Home')next=0;else if(e.key==='End')next=2;else return;e.preventDefault();selectStep(next);$(`#step-${next}`).focus();});});
['method-video','env-video'].forEach(id=>{const v=document.getElementById(id);v.addEventListener('play',()=>{pauseFilm();['method-video','env-video'].filter(x=>x!==id).forEach(x=>document.getElementById(x).pause());});new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)v.pause();},{threshold:0.01}).observe(v);});

let data, scope='all', family='all', descending=true;
function renderRanking(){
 const envs=selectEnvironments(data,scope,family);const rows=summarize(data,scope,family,descending);
 const plannerCount=envs.filter(e=>e.results.planner!==null).length;
 $('#scope-note').textContent=`${envs.length} environments · ${family==='all'?'All five families':family}. ${scope==='all' && plannerCount!==envs.length?`Planner covers ${plannerCount}/${envs.length}; its partial-coverage score is unranked. Select the planner subset for a matched comparison.`:'All methods are compared on the same environments.'}`;
 $('#ranking-body').innerHTML=rows.map(r=>`<tr class="${r.kind==='reference'?'reference':r.rank===1?'winner':''}"><td class="rank">${r.rank===null?'—':String(r.rank).padStart(2,'0')}</td><td><span class="method-name">${esc(r.name)}</span><span class="backend">${esc(r.backend)}</span>${r.kind==='reference'?'<span class="reference-label">Source-access reference</span>':''}</td><td><span class="access-tag">${esc(r.access)}</span></td><td class="coverage">${r.count} / ${r.total}</td><td class="score-cell"><div class="score-flex"><span class="score-track"><span class="score-fill" style="--w:${r.mean===null?0:r.mean*100}%;--c:${r.color}"></span></span><span class="score-number">${r.mean===null?'—':percent(r.mean)}</span></div></td></tr>`).join('');
 $('#sort-score').innerHTML=`Mean success <span aria-hidden="true">${descending?'↓':'↑'}</span>`;$('#sort-score').closest('th').setAttribute('aria-sort',descending?'descending':'ascending');
}
$$('[data-scope]').forEach(b=>b.addEventListener('click',()=>{scope=b.dataset.scope;$$('[data-scope]').forEach(x=>{const active=x===b;x.classList.toggle('active',active);x.setAttribute('aria-pressed',String(active));});if(data)renderRanking();}));
$('#family-filter').addEventListener('change',e=>{family=e.target.value;if(data)renderRanking();});$('#sort-score').addEventListener('click',()=>{descending=!descending;if(data)renderRanking();});
function selectEnvironment(id){
 const env=data.environments.find(e=>e.id===id);if(!env)return;
 $$('[data-env]').forEach(b=>{b.classList.toggle('active',b.dataset.env===id);b.setAttribute('aria-pressed',String(b.dataset.env===id));});
 $('#env-family').textContent=env.family;$('#env-name').textContent=env.name;const v=$('#env-video');v.pause();v.poster=env.poster;v.src=env.video;v.load();
 const order=['claude','codex','genplan','oneshot','planner','source'];
 $('#env-bars').innerHTML=order.map(id=>{const m=data.methods.find(x=>x.id===id),r=env.results[id];const name={claude:'Agentic · Claude Code',codex:'Agentic · Codex',genplan:'LLMGenPlan',oneshot:'One-shot',planner:'Planner',source:'Agentic + source'}[id];return `<div class="env-row"><strong>${name}</strong><span class="score-track"><span class="score-fill" style="--w:${r?r.mean*100:0}%;--c:${m.color}"></span></span><span class="env-value">${r?`${whole(r.mean)} <small>[${whole(r.min)}–${whole(r.max)}]</small>`:'Not available'}</span></div>`;}).join('');
}
function renderEnvironmentList(){let last='';$('#env-list').innerHTML=data.environments.map(e=>{const heading=e.family!==last?`<p class="env-group-title">${e.family}</p>`:'';last=e.family;return `${heading}<button data-env="${e.id}" aria-pressed="false">${e.name}<span aria-hidden="true">↗</span></button>`;}).join('');$$('[data-env]').forEach(b=>b.addEventListener('click',()=>selectEnvironment(b.dataset.env)));selectEnvironment('Tossing3D');}
$$('[data-select-env]').forEach(b=>b.addEventListener('click',()=>{if(!data)return;selectEnvironment(b.dataset.selectEnv);$('.explorer').scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'start'});}));

let gallery=[],category='all';const dialog=$('#gallery-dialog');
function provenance(g){return `${g.method} · ${g.backend} · ${g.setting} · seed ${g.seed} · episode ${g.episode}`;}
function openGallery(id){const g=gallery.find(g=>g.id===id);if(!g)return;const v=$('#dialog-video');v.src=`film/assets/clips/${g.file}.mp4`;v.poster=`assets/posters/${g.file}.jpg`;$('#dialog-category').textContent=`${g.category} / ${g.environment}`;$('#dialog-title').textContent=g.title;$('#dialog-description').textContent=g.description;$('#dialog-provenance').textContent=`${provenance(g)}. Source clip: ${g.file}.mp4 · playback 1×.`;$('#method-video').pause();$('#env-video').pause();pauseFilm();dialog.showModal();document.body.style.overflow='hidden';v.play().catch(()=>{});}
function renderGallery(){const items=gallery.filter(g=>category==='all'||g.category===category);$('#gallery-grid').innerHTML=items.map(g=>`<article class="gallery-card"><button class="gallery-thumb" data-gallery="${g.id}" aria-label="Play: ${esc(g.title)}"><img loading="lazy" src="assets/posters/${g.file}.jpg" alt="${esc(g.environment)} example rollout" width="720" height="540"><span class="small-play" aria-hidden="true">▶</span></button><p class="gallery-type">${esc(g.category)} / ${esc(g.environment)}</p><h3>${esc(g.title)}</h3><p class="description">${esc(g.description)}</p><p class="provenance"><strong>${esc(g.method)} · ${esc(g.backend)}</strong><br>${esc(g.setting)} · seed ${g.seed} · episode ${g.episode}</p></article>`).join('');$$('[data-gallery]').forEach(b=>b.addEventListener('click',()=>openGallery(b.dataset.gallery)));}
$$('[data-category]').forEach(b=>b.addEventListener('click',()=>{category=b.dataset.category;$$('[data-category]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});renderGallery();}));
$('.dialog-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});dialog.addEventListener('close',()=>{$('#dialog-video').pause();$('#dialog-video').removeAttribute('src');$('#dialog-video').load();document.body.style.overflow='';});
async function loadJSON(path){const res=await fetch(path);if(!res.ok)throw new Error(`Cannot load ${path}: ${res.status}`);return res.json();}
loadJSON('data/benchmark.json').then(result=>{data=result;renderRanking();renderEnvironmentList();}).catch(error=>{console.error(error);$('#scope-note').textContent='The interactive results could not load. Please download the CSV or read Tables I–II in the paper.';});
loadJSON('data/gallery.json').then(result=>{gallery=result;renderGallery();}).catch(error=>{console.error(error);$('#gallery-grid').innerHTML='<p>The gallery could not load. <a href="film/">Open the project film instead ↗</a></p>';});
