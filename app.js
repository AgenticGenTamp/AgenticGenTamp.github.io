import {selectEnvironments, summarize} from './benchmark.js';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const percent = n => `${(n * 100).toFixed(1)}%`;
const whole = n => `${Math.round(n * 100)}%`;
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hero = $('#hero-video');
let heroWanted = !reduced.matches && !navigator.connection?.saveData;
function updateMotionButton() { $('#motion-toggle').textContent = hero.paused ? 'Play background' : 'Pause background'; $('#motion-toggle').setAttribute('aria-label', hero.paused ? 'Play background video' : 'Pause background video'); }
hero.addEventListener('play', updateMotionButton);hero.addEventListener('pause', updateMotionButton);
$('#motion-toggle').addEventListener('click', () => { heroWanted = hero.paused; if (heroWanted) hero.play().catch(updateMotionButton); else hero.pause(); });
new IntersectionObserver(entries => { const visible = entries[0].isIntersecting; if (visible && heroWanted && !document.hidden) hero.play().catch(updateMotionButton); else hero.pause(); }, {threshold:0.05}).observe(hero);
reduced.addEventListener('change', () => { if (reduced.matches) {heroWanted=false;hero.pause();galleryVisible.forEach(pauseGalleryVideo);} else galleryVisible.forEach(playGalleryVideo); });
document.addEventListener('visibilitychange',()=>{ if(document.hidden){hero.pause();$('#env-video').pause();pausePolicyExamples();galleryVisible.forEach(pauseGalleryVideo);pauseFilm();}else {if(heroWanted && $('#top').getBoundingClientRect().bottom>0)hero.play().catch(()=>{});galleryVisible.forEach(playGalleryVideo);} });
let scrolled = false;
function navState(){const next=window.scrollY>100;if(next!==scrolled){$('#nav').classList.toggle('sticky',next);scrolled=next;}}
window.addEventListener('scroll',navState,{passive:true});navState();

// Native video keeps playback controls in a fixed, responsive frame.
const projectVideo = $('#project-video');
const envVideo = $('#env-video');
function pauseFilm() { projectVideo.pause(); }
projectVideo.addEventListener('play', () => {envVideo.pause();pausePolicyExamples();galleryVisible.forEach(pauseGalleryVideo);});
projectVideo.addEventListener('pause', () => galleryVisible.forEach(playGalleryVideo));
envVideo.addEventListener('play', () => {pauseFilm();pausePolicyExamples();});
new IntersectionObserver(entries => {if (!entries[0].isIntersecting) pauseFilm();}, {threshold:0.01}).observe(projectVideo);
projectVideo.addEventListener('error', () => {
  const message = document.createElement('p');
  message.className = 'video-error';
  message.textContent = 'The video could not load. Please reload the page and try again.';
  if (!$('#video .video-error')) projectVideo.after(message);
});
new IntersectionObserver(entries => {if (!entries[0].isIntersecting) envVideo.pause();}, {threshold:0.01}).observe(envVideo);

let data, descriptions, policyExamples, selectedEnvironment, descriptionLoadFailed=false;
let scope='all', family='all', descending=true;
function renderRanking(){
 const envs=selectEnvironments(data,scope,family);const rows=summarize(data,scope,family,descending);
 const plannerCount=envs.filter(e=>e.results.planner!==null).length;
 $('#scope-note').textContent=`${envs.length} environments · ${family==='all'?'All five families':family}. ${scope==='all' && plannerCount!==envs.length?`Planner covers ${plannerCount}/${envs.length}; its mean uses only those environments. Select the planner subset for a matched comparison.`:'All methods are compared on the same environments.'}`;
 $('#ranking-body').innerHTML=rows.map(r=>`<tr class="${r.kind==='reference'?'reference':''}"><td><span class="method-name">${esc(r.name)}</span><span class="backend">${esc(r.backend)}</span>${r.kind==='reference'?'<span class="reference-label">Source-access reference</span>':''}</td><td><span class="access-tag">${esc(r.id==='source'?'+ source':r.access==='Black box'?'Main setting':r.access)}</span></td><td class="coverage">${r.count} / ${r.total}</td><td class="score-cell"><div class="score-flex"><span class="score-track"><span class="score-fill" style="--w:${r.mean===null?0:r.mean*100}%;--c:${r.color}"></span></span><span class="score-number">${r.mean===null?'—':percent(r.mean)}</span></div></td></tr>`).join('');
 $('#sort-score').innerHTML=`Mean success <span aria-hidden="true">${descending?'↓':'↑'}</span>`;$('#sort-score').closest('th').setAttribute('aria-sort',descending?'descending':'ascending');
}
$$('[data-scope]').forEach(b=>b.addEventListener('click',()=>{scope=b.dataset.scope;$$('[data-scope]').forEach(x=>{const active=x===b;x.classList.toggle('active',active);x.setAttribute('aria-pressed',String(active));});if(data)renderRanking();}));
$('#family-filter').addEventListener('change',e=>{family=e.target.value;if(data)renderRanking();});$('#sort-score').addEventListener('click',()=>{descending=!descending;if(data)renderRanking();});
function renderEnvironmentDescription() {
 const panel=$('#env-description-content');
 const raw=$('#env-description-raw');
 if(!selectedEnvironment)return;
 raw.href=`data/environment-descriptions/${encodeURIComponent(selectedEnvironment)}.md`;
 const entry=descriptions?.environments.find(e=>e.id===selectedEnvironment);
 panel.setAttribute('aria-label', `Agent description: ${$('#env-name').textContent}`);
 if(entry){
   // HTML is generated from the checked-in Markdown with raw HTML disabled.
   panel.innerHTML=entry.html;
   raw.href=entry.rawFile;
   panel.setAttribute('aria-busy','false');
 }else if(descriptionLoadFailed || descriptions){
   panel.textContent='The description could not load. Use the Original Markdown link above to read the archived text.';
   panel.setAttribute('aria-busy','false');
 }else{
   panel.textContent='Loading the environment description…';
   panel.setAttribute('aria-busy','true');
 }
 panel.scrollTop=0;
 panel.scrollLeft=0;
}
function selectEnvironment(id){
 const env=data.environments.find(e=>e.id===id);if(!env)return;
 $$('[data-env]').forEach(b=>{b.classList.toggle('active',b.dataset.env===id);b.setAttribute('aria-pressed',String(b.dataset.env===id));});
 $('#env-family').textContent=env.family;$('#env-name').textContent=env.name;selectedEnvironment=id;renderEnvironmentDescription();renderPolicyExamples();const v=$('#env-video');v.pause();v.poster=env.poster;v.src=env.video;v.load();
 const order=['claude','codex','genplan','oneshot','planner','source'];
 $('#env-bars').innerHTML=order.map(id=>{const m=data.methods.find(x=>x.id===id),r=env.results[id];const name={claude:'Agentic · Claude Code',codex:'Agentic · Codex',genplan:'LLMGenPlan',oneshot:'One-shot',planner:'Planner',source:'Agentic + source'}[id];return `<div class="env-row"><strong>${name}</strong><span class="score-track"><span class="score-fill" style="--w:${r?r.mean*100:0}%;--c:${m.color}"></span></span><span class="env-value">${r?`${whole(r.mean)} <small>[${whole(r.min)}–${whole(r.max)}]</small>`:'Not available'}</span></div>`;}).join('');
}
function renderEnvironmentList(){let last='';$('#env-list').innerHTML=data.environments.map(e=>{const heading=e.family!==last?`<p class="env-group-title">${e.family}</p>`:'';last=e.family;return `${heading}<button data-env="${e.id}" aria-pressed="false">${e.name}<span aria-hidden="true">↗</span></button>`;}).join('');$$('[data-env]').forEach(b=>b.addEventListener('click',()=>selectEnvironment(b.dataset.env)));selectEnvironment('StickButton2D');markPolicyExamples();}
$$('[data-select-env]').forEach(b=>b.addEventListener('click',()=>{if(!data)return;selectEnvironment(b.dataset.selectEnv);$('.explorer').scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'start'});}));

function markPolicyExamples() {
  if(!policyExamples)return;
  $$('[data-env]').forEach(button=>{
    if(policyExamples.environments.some(e=>e.id===button.dataset.env)){
      const marker=button.querySelector('span');marker.textContent='3 videos';marker.className='policy-marker';
      button.setAttribute('aria-label',`${data.environments.find(e=>e.id===button.dataset.env).name}, three policy videos`);
    }
  });
}
function pausePolicyExamples() { $$('.policy-video').forEach(video => video.pause()); }
new IntersectionObserver(entries => {if(!entries[0].isIntersecting)pausePolicyExamples();},{threshold:0.01}).observe($('#policy-comparison'));
function renderPolicyExamples() {
  pausePolicyExamples();
  $$('.policy-video').forEach(video => {video.removeAttribute('src');video.load();});
  const example=policyExamples?.environments.find(e=>e.id===selectedEnvironment);
  $('#policy-comparison').hidden=!example;
  $('#policy-grid').replaceChildren();
  if(!example)return;
  $('#policy-comparison-note').textContent='The frozen programs act on the same held-out instance. These selected examples illustrate behavioral differences, not average performance, and are not supplied to any agent. All clips use the same action playback rate; video duration is not computation time.';
  $('#policy-grid').innerHTML=example.videos.map(clip=>`<figure class="policy-card">
    <h5>${esc(clip.label)}</h5><p>${esc(clip.setting)}</p>
    <video class="policy-video" src="${esc(clip.video)}" poster="${esc(clip.poster)}" controls muted playsinline preload="none" aria-label="${esc(clip.label)} policy in ${esc($('#env-name').textContent)}"></video>
    <figcaption><strong>${clip.solved?'Success':'Failure'}</strong> · ${clip.steps} actions</figcaption>
  </figure>`).join('');
  $$('.policy-video').forEach(video => {
    video.muted=true;
    video.addEventListener('play',()=>{pauseFilm();envVideo.pause();galleryVisible.forEach(pauseGalleryVideo);});
  });
}
$('#play-policies').addEventListener('click',()=>{
  $$('.policy-video').forEach(video=>{video.currentTime=0;video.play().catch(()=>{});});
});

let gallery=[];
let gallerySpeed=2;
const galleryVisible = new Set();
const galleryPausedByUser = new WeakSet();
const galleryAutomaticPauses = new WeakSet();
function pauseGalleryVideo(video) {
  if (!video.paused) {
    galleryAutomaticPauses.add(video);
    video.pause();
  }
}
function playGalleryVideo(video) {
  if (document.hidden || !projectVideo.paused || $$('.policy-video').some(v=>!v.paused) || reduced.matches || navigator.connection?.saveData || galleryPausedByUser.has(video) || !galleryVisible.has(video)) return;
  video.play().then(() => {
    // A play request may settle after a scroll, tab switch, or another video starts.
    if (!galleryVisible.has(video) || document.hidden || reduced.matches || !projectVideo.paused || $$('.policy-video').some(v=>!v.paused)) pauseGalleryVideo(video);
  }).catch(() => {}); // Native controls remain available if autoplay is blocked.
}
const galleryObserver = new IntersectionObserver(entries => {
  entries.forEach(({target:video,isIntersecting,intersectionRatio}) => {
    if (isIntersecting && intersectionRatio >= 0.2) {
      galleryVisible.add(video);
      if (!video.getAttribute('src')) {
        video.src = video.dataset.src;
        video.load();
      }
      playGalleryVideo(video);
    } else {
      galleryVisible.delete(video);
      pauseGalleryVideo(video);
    }
  });
}, {threshold:[0,0.2]});
function renderGallery() {
  galleryObserver.disconnect();
  galleryVisible.clear();
  $$('.gallery-video').forEach(video => {pauseGalleryVideo(video);video.removeAttribute('src');video.load();});
  $('#gallery-grid').innerHTML=gallery.map(g=>`<article class="gallery-card">
    <div class="gallery-thumb"><video class="gallery-video" data-gallery="${g.id}" data-src="film/assets/clips/${g.file}.mp4" poster="assets/posters/${g.file}.jpg" muted loop playsinline controls preload="none" aria-labelledby="gallery-title-${g.id}" aria-describedby="gallery-provenance-${g.id}"></video></div>
    <p class="gallery-type">${esc(g.environment)}</p>
    <h3 id="gallery-title-${g.id}">${esc(g.title)}</h3>
    <p class="description">${esc(g.description)}</p>
    <p class="provenance" id="gallery-provenance-${g.id}"><strong>${esc(g.backend)}</strong> · ${esc(g.setting)}</p>
  </article>`).join('');
  $$('.gallery-video').forEach(video => {
    video.muted = true;
    video.defaultPlaybackRate = gallerySpeed;
    video.playbackRate = gallerySpeed;
    video.addEventListener('loadedmetadata', () => {video.playbackRate=gallerySpeed;});
    video.addEventListener('play', () => galleryPausedByUser.delete(video));
    video.addEventListener('pause', () => {
      if (galleryAutomaticPauses.delete(video)) return;
      if (galleryVisible.has(video) && !document.hidden && !reduced.matches) galleryPausedByUser.add(video);
    });
    galleryObserver.observe(video);
  });
}
$('#gallery-speed').addEventListener('change', event => {
  gallerySpeed=Number(event.target.value);
  $$('.gallery-video').forEach(video => {video.defaultPlaybackRate=gallerySpeed;video.playbackRate=gallerySpeed;});
});
async function loadJSON(path){const res=await fetch(path);if(!res.ok)throw new Error(`Cannot load ${path}: ${res.status}`);return res.json();}
loadJSON('data/benchmark.json').then(result=>{data=result;renderRanking();renderEnvironmentList();}).catch(error=>{console.error(error);$('#scope-note').textContent='The interactive results could not load. Please download the CSV or read Tables I–II in the paper.';});
loadJSON('data/gallery.json').then(result=>{gallery=result;renderGallery();}).catch(error=>{console.error(error);$('#gallery-grid').innerHTML='<p>The gallery could not load. <a href="assets/project-video.mp4">Watch the project video ↗</a></p>';});

loadJSON('data/environment-descriptions.json?v=descriptions-5').then(result=>{descriptions=result;renderEnvironmentDescription();}).catch(error=>{console.error(error);descriptionLoadFailed=true;renderEnvironmentDescription();});

loadJSON('data/policy-examples.json?v=results-8').then(result=>{policyExamples=result;renderPolicyExamples();markPolicyExamples();}).catch(error=>{console.error(error);$('#policy-comparison').hidden=true;});
