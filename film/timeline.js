// Time-driven presentation. One clock drives scene visibility, reveals, clips and diagrams.

const stage = document.getElementById('stage');
const captionsEl = document.getElementById('captions');
const scrub = document.getElementById('scrub');
const clockEl = document.getElementById('clock');
const sceneNameEl = document.getElementById('scene-name');
const playBtn = document.getElementById('play');
const themeSel = document.getElementById('theme');
const capsBox = document.getElementById('caps');
const TOTAL = SCENES[SCENES.length - 1].end;
scrub.max = TOTAL;
const THEMES = ['poster', 'journal', 'mono'];

const params = new URLSearchParams(location.search);
let t = parseFloat(params.get('t') || '0');
let playing = (params.has('record') || params.has('autoplay')) && !params.has('paused');
let holding = false;   // the clock is waiting for a clip to finish
let elapsed = 0;       // real seconds spent playing, holds included
if (params.has('record')) document.body.classList.add('record');

function setTheme(name) {
  if (!THEMES.includes(name)) return;
  document.body.dataset.theme = name;
  themeSel.value = name;
}
setTheme(params.get('theme') || 'poster');
function setCaptions(on) { document.body.classList.toggle('nocaps', !on); capsBox.checked = on; }
setCaptions(params.get('captions') === '1');

// Fit the fixed 1280x720 stage to the window.
function fit() {
  const wrap = document.getElementById('stage-wrap');
  const s = Math.min(wrap.clientWidth / 1280, wrap.clientHeight / 720);
  stage.style.transform = `scale(${s})`;
}
window.addEventListener('resize', fit);
fit();

// Results grid: 28 environments, seven per row, in the paper's family order.
const gridEl = document.getElementById('grid');
let family = '';
GRID.forEach(e => {
  if (e.fam) family = e.fam;
  const d = document.createElement('div');
  d.className = 'gt';
  const img = e.img
    ? `<video class="clip" data-src="assets/clips/grid/${e.img}.mp4" data-free="1" muted loop playsinline preload="auto"></video>`
    : `<div class="ph" title="episode to render"></div>`;
  const ok = e.agent >= 0.95 ? '<span class="ok" title="95% or above"></span>' : '';
  d.innerHTML = `<div class="th">${img}${ok}</div><div class="nm"><small>${family}</small>${e.name}</div>`;
  gridEl.appendChild(d);
});
const gridTiles = Array.from(gridEl.querySelectorAll('.gt'));

// Protocol scenes: each version has its own terminal and evaluation counter.
const protocolEl = document.getElementById(PROTOCOL_ID);

// Clips load lazily: only the active scene and its neighbours hold a source, so the
// browser's cap on concurrent media loads never starves the clip that is on screen.
function loadScenes(activeIndex) {
  SCENES.forEach((s, i) => {
    const near = Math.abs(i - activeIndex) <= 1;
    document.getElementById(s.id).querySelectorAll('video.clip').forEach(v => {
      if (near && !v.getAttribute('src')) { v.src = v.dataset.src; v.load(); }
      if (!near && v.getAttribute('src')) { v.removeAttribute('src'); v.load(); }
    });
  });
}

// Cues: every moment where something new appears. Arrow keys step through them.
const CUES = (() => {
  const all = [...new Set([
    ...SCENES.map(s => s.start),
    // Sections that are not on the timeline (the other protocol version) contribute nothing.
    ...Array.from(document.querySelectorAll('.scene .reveal:not(.minor)')).flatMap(r => {
      const s = SCENES.find(s => s.id === r.closest('.scene').id);
      if (!s) return [];
      return [s.start + +r.dataset.at, ...(r.dataset.until ? [s.start + +r.dataset.until] : [])];
    }),
    ...Array.from(document.querySelectorAll('.beat')).flatMap(b => { const s = SCENES.find(s => s.id === b.closest('.scene').id); return s ? [s.start + +b.dataset.from] : []; }),
    ...EXTRA_CUES,
  ])].sort((a, b) => a - b);
  // Moments less than a second apart belong to one segment; scene starts always stay.
  const starts = new Set(SCENES.map(s => s.start));
  const out = [];
  for (const c of all) if (starts.has(c) || out.length === 0 || c - out[out.length - 1] >= 1) out.push(c);
  return out;
})();

// Scaling plots as SVG paths with pathLength=1, drawn by animating stroke-dashoffset.
function buildPlot(svg, key, yMax, title, yFmt) {
  const L = 52, R = 16, T = 30, B = 44, W = 400, H = 260;
  const x = i => L + (i / (SCALING.levels.length - 1)) * (W - L - R);
  const y = v => T + (1 - v / yMax) * (H - T - B);
  const path = arr => arr.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const ticks = [0, 0.5, 1].map(f => f * yMax);
  svg.innerHTML = `
    <text class="ttl" x="${L}" y="18">${title}</text>
    <line class="axis" x1="${L}" y1="${y(0)}" x2="${W - R}" y2="${y(0)}"/>
    <line class="axis" x1="${L}" y1="${y(0)}" x2="${L}" y2="${T}"/>
    ${ticks.map(v => `<text class="tick" x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${yFmt(v)}</text>`).join('')}
    <text class="alab" x="${L}" y="${H - 14}">fewer objects</text>
    <text class="alab" x="${W - R}" y="${H - 14}" text-anchor="end">more objects</text>
    <path class="line-plot genplan" pathLength="1" d="${path(SCALING.genplan[key])}"/>
    <path class="line-plot planner" pathLength="1" d="${path(SCALING.planner[key])}"/>
    <path class="line-plot program" pathLength="1" d="${path(SCALING.program[key])}"/>`;
}
buildPlot(document.getElementById('plot-success'), 'success', 1, 'Success rate', v => v.toFixed(1));
buildPlot(document.getElementById('plot-time'), 'time', 50, 'Computation per instance (s)', v => v.toFixed(0));
const drawn = Array.from(document.querySelectorAll('#plots path.line-plot'));

// Family means from the grid data: agent and genplan over every environment, planner where one exists.
const FAMILIES = [];
GRID.forEach(e => { if (e.fam) FAMILIES.push({ fam: e.fam, rows: [] }); FAMILIES[FAMILIES.length - 1].rows.push(e); });
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
document.getElementById('fambars').innerHTML = FAMILIES.map(f => {
  const a = mean(f.rows.map(r => r.agent)), g = mean(f.rows.map(r => r.genplan));
  const pl = f.rows.filter(r => r.planner !== null).map(r => r.planner);
  const p = pl.length ? mean(pl) : null;
  const bar = (k, v) => v === null ? '<div class="vb none"></div>' : `<div class="vb ${k}"><i style="height:${(v * 100).toFixed(0)}%"></i><b>${Math.round(v * 100)}</b></div>`;
  return `<div class="fg">${bar('agent', a)}${bar('planner', p)}${bar('genplan', g)}<span>${f.fam}</span></div>`;
}).join('');
document.getElementById('effbars').innerHTML = EFFICIENCY.map((e, i) => `<div class="hb${i ? ' src' : ''}"><span>${e.name}</span><i style="width:${(e.ms / 30 * 100).toFixed(0)}%"></i><b>${e.ms.toFixed(1)} ms</b></div>`).join('');

// Per-scene hooks receive the local time within the scene (or -1 when inactive).
const hookCode = document.getElementById('hookcode');

// Minimal Python syntax colouring for typed code. Works on partial text.
const KEYWORDS = /\b(def|return|if|else|elif|for|while|in|not|and|or|None|True|False|import|from|class|pass|break|continue|lambda|with|as|is)\b/g;
function highlight(src) {
  const esc = src.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const out = [];
  const re = /(#[^\n]*)|('[^'\n]*'?|"[^"\n]*"?)|(\bdef\s+)(\w+)|(\bself\b)|(\b\d+(?:\.\d+)?\b)/g;
  let i = 0, m;
  const plain = s => s.replace(KEYWORDS, '<span class="kw">$1</span>');
  while ((m = re.exec(esc))) {
    out.push(plain(esc.slice(i, m.index)));
    if (m[1]) out.push(`<span class="cm">${m[1]}</span>`);
    else if (m[2]) out.push(`<span class="str">${m[2]}</span>`);
    else if (m[3]) out.push(`<span class="kw">${m[3].trim()}</span> <span class="fn">${m[4]}</span>`);
    else if (m[5]) out.push(`<span class="slf">${m[5]}</span>`);
    else if (m[6]) out.push(`<span class="num">${m[6]}</span>`);
    i = re.lastIndex;
  }
  out.push(plain(esc.slice(i)));
  return out.join('');
}
// Fix the panel to the size of the finished text so it does not grow while typing.
hookCode.innerHTML = highlight(HOOK_CODE);
hookCode.style.width = `${hookCode.offsetWidth}px`;
hookCode.style.height = `${hookCode.offsetHeight}px`;
hookCode.textContent = '';

const problemEl = document.getElementById('problem');
const problemOuts = problemEl.querySelectorAll('.out');
const OUT_WINDOWS = [[[12.3, 18], [22.3, 26], [31, 45]], [[12.5, 18], [22.3, 26], [31, 45]], [[12.7, 18], [22.3, 26], [31, 45]]];
{
  const set = PROBLEM_SETS[new URLSearchParams(location.search).get('env')] || PROBLEM_SETS.obstruction3d;
  problemEl.querySelectorAll('.inp').forEach((el, i) => { el.querySelector('video').dataset.src = set.clips[i]; el.querySelector('.k').textContent = set.caps[i]; });
  problemOuts.forEach((el, i) => { el.querySelector('video').dataset.src = set.clips[i]; });
  problemEl.querySelector('.envloop').dataset.src = set.loop;
  problemEl.classList.toggle('r43', set.aspect === '4/3');
}

const studyEl = document.getElementById('study');

const hooks = {
  study(lt) { studyEl.classList.toggle('wide', lt >= 1); },
  problem(lt) {
    problemEl.classList.toggle('side', lt >= 6.5);
    problemOuts.forEach((el, i) => el.classList.toggle('on', OUT_WINDOWS[i].some(([a, b]) => lt >= a && lt < b)));
    problemEl.querySelectorAll('.top .eng, .top .abs').forEach(n => n.classList.toggle('gone', lt >= 27.5));
    problemEl.querySelector('.agent2').classList.toggle('think', lt >= 27.5 && lt < 30.3);
  },
  ...Object.fromEntries([['protocol', TERM_SHELF, 40], ['protocol-stick', TERM, 34]].map(([id, script, evalAt]) => {
    const sec = document.getElementById(id);
    const termEl = sec.querySelector('.term'), scoreEl = sec.querySelector('.evalscore');
    const checks = sec.querySelector('.checks'); checks.innerHTML = '<i></i>'.repeat(100);
    const cells = Array.from(checks.children), times = EVAL_TIMES[id];
    const mosaic = sec.querySelector('.evalbox video'); const rate = parseFloat(mosaic.dataset.speed || '1'), t0 = parseFloat(mosaic.dataset.starts);
    const esc = x => x.replace(/&/g, '&amp;').replace(/</g, '&lt;');
    return [id, lt => {
      // Terminal: every event at or before lt contributes a line; typed code grows with time.
      let lines = [];
      for (const ev of script) {
        if (lt < ev.at) break;
        if (ev.type === 'clear') { lines = []; continue; }
        if (ev.type === 'type') lines.push(`<div class="code">${highlight(ev.text.slice(0, Math.floor((lt - ev.at) * (ev.cps || 60))))}</div>`);
        else lines.push(`<div class="${ev.type}">${esc(ev.text).replace(/\b(True|False)\b/gi, m => `<span class="${m.toLowerCase() === 'true' ? 'ok' : 'bad'}">${m}</span>`)}</div>`);
      }
      termEl.innerHTML = lines.join('');
      termEl.scrollTop = termEl.scrollHeight;
      sec.classList.toggle('frozen', lt >= evalAt);
      sec.classList.toggle('off', lt >= evalAt);
      // Evaluation: a check appears on each episode as it finishes in the mosaic; the count follows.
      let n = 0;
      cells.forEach((c, i) => { const on = times[i] !== null && lt >= t0 + times[i] / rate; c.classList.toggle('ok', on); if (on) n++; });
      scoreEl.textContent = String(n);
    }];
  })),
  results(lt) {
    gridTiles.forEach((tile, i) => tile.classList.toggle('on', lt >= i * 0.06));
    gridEl.classList.toggle('scored', lt >= 3);
    gridEl.classList.toggle('away', lt >= 7);
    const p = Math.max(0, Math.min(1, (lt - 7.5) / 1.5));
    // Lines draw along their length over 1.5 s and then stay fully drawn.
    drawn.forEach(path => { path.style.strokeDasharray = '1'; path.style.strokeDashoffset = String(1 - p); });
  },
};

function activeScene() {
  return SCENES.find(s => t >= s.start && t < s.end) || SCENES[SCENES.length - 1];
}

// A clip is finished when it has played to its end and then held its last frame for
// data-tail seconds (default TAIL), so an episode does not cut away the instant it ends.
// Unloaded clips count as finished after a grace period so a missing file cannot stall.
const TAIL = 1.5;
// data-end cuts a clip short at that many seconds of its own time.
function clipEnd(v) { return v.dataset.end ? Math.min(parseFloat(v.dataset.end), v.duration) : v.duration; }
function clipFinished(v) {
  if (!(v.duration > 0)) return (v._waitedSince ??= performance.now()) < performance.now() - 8000;
  const atEnd = v.ended || v.currentTime >= clipEnd(v) - 0.08;
  if (!atEnd) { v._endedAt = null; return false; }
  v._endedAt ??= performance.now();
  const tail = v.dataset.tail !== undefined ? parseFloat(v.dataset.tail) : TAIL;
  return performance.now() - v._endedAt >= tail * 1000;
}

// The container of a clip is its beat, or its scene when it has no beat.
function clipWindow(v, scene) {
  const beat = v.closest('.beat');
  return beat
    ? { start: scene.start + +beat.dataset.from, end: scene.start + +beat.dataset.to }
    : { start: scene.start, end: scene.end };
}

// Gate: the earliest container end among the clips now playing that are not yet finished.
function gate() {
  const sc = activeScene();
  const el = document.getElementById(sc.id);
  let end = Infinity, done = true;
  el.querySelectorAll('video.clip:not([data-free])').forEach(v => {
    if (v.dataset.wait === '0' || v.dataset.still) return;
    const w = clipWindow(v, sc);
    if (t < w.start || t >= w.end) return;
    end = Math.min(end, w.end);
    if (!clipFinished(v)) done = false;
  });
  return { end, done };
}

// A gated clip that is playing and has not reached its end. A segment stop waits for it,
// so stepping with the arrow keys never freezes an episode halfway through.
function midClip() {
  const sc = activeScene();
  const el = document.getElementById(sc.id);
  return Array.from(el.querySelectorAll('video.clip:not([data-free])')).some(v => {
    if (v.dataset.wait === '0' || v.dataset.still || v.paused || !(v.duration > 0)) return false;
    const w = clipWindow(v, sc);
    return t >= w.start && t < w.end && !(v.ended || v.currentTime >= clipEnd(v) - 0.08);
  });
}

function syncClip(v, sceneStart, active, noSeek) {
  if (!active) { if (!v.paused) v.pause(); return; }
  const shouldPlay = playing;
  if (v.dataset.still) {
    // A still shows the first frame of its clip and never plays.
    if (!v.paused) v.pause();
    // A paused video paints nothing until a seek completes, so nudge it once after metadata arrives.
    if (v.duration > 0 && !v._primed) { v._primed = true; v.currentTime = 0.01; }
    return;
  }
  if (v.dataset.free) {
    // Background loops: play with the clock, never seek.
    if (shouldPlay && v.paused) v.play().catch(() => {});
    if (!shouldPlay && !v.paused) v.pause();
    return;
  }
  const rate = parseFloat(v.dataset.speed || '1');
  const offset = parseFloat(v.dataset.offset || '0');
  v.playbackRate = rate;
  const dur = v.duration ? clipEnd(v) : Infinity;
  // data-starts lists container-local times at which the clip (re)starts from data-offset.
  const lt = t - sceneStart;
  let from = null;
  for (const s of (v.dataset.starts || '0').split(',').map(Number)) if (lt >= s) from = s;
  const before = from === null;
  const raw = before ? offset : offset + (lt - from) * rate;
  const pastEnd = raw >= dur - 0.05;
  const want = Math.min(raw, dur - 0.05);
  // Seek on scrubs and large drift only; small drift would stutter the clip.
  const tolerance = shouldPlay ? 0.75 : 0.05;
  // Never re-issue a seek while one is in flight; each new seek cancels the last fetch.
  // A lagging clip whose slot has run out plays on to its real end; the gate waits for it.
  const seekable = !noSeek && !v.seeking && !(shouldPlay && pastEnd);
  if (seekable && Math.abs(v.currentTime - want) > tolerance) v.currentTime = want;
  // Only the clip's own position says it is over, so the closing frames are never skipped.
  const atEnd = v.ended || v.currentTime >= dur - 0.08;
  // A clip that has ended stays on its last frame; play() would restart it from zero.
  if (shouldPlay && !before && !atEnd && v.paused && !v.ended) v.play().catch(() => {});
  if ((!shouldPlay || before || atEnd) && !v.paused) v.pause();
}

function render() {
  const sc = activeScene();
  loadScenes(SCENES.indexOf(sc));
  SCENES.forEach(s => {
    const el = document.getElementById(s.id);
    const active = s === sc;
    el.classList.toggle('active', active);
    const lt = t - s.start;
    el.querySelectorAll('.beat').forEach(b => {
      b.classList.toggle('on', active && lt >= +b.dataset.from && lt < +b.dataset.to);
    });
    el.querySelectorAll('.reveal').forEach(r => {
      const until = r.dataset.until ? +r.dataset.until : Infinity;
      r.classList.toggle('on', active && lt >= +r.dataset.at && lt < until);
    });
    el.querySelectorAll('video.clip').forEach(v => {
      const w = clipWindow(v, s);
      const inWindow = t >= w.start && t < w.end;
      // While the clock holds for this clip, let it run free instead of pinning it to the clock.
      syncClip(v, w.start, active && inWindow, holding && inWindow);
    });
    if (hooks[s.id]) hooks[s.id](active ? lt : -1);
  });
  const cap = CAPTIONS.find(c => t >= c[0] && t < c[1]);
  captionsEl.innerHTML = cap ? `<span>${cap[2]}</span>` : '';
  scrub.value = t;
  clockEl.textContent = `${t.toFixed(1)} s · real ${elapsed.toFixed(1)} s${holding ? ' · waiting for clip' : ''}`;
  sceneNameEl.textContent = sc.id;
  playBtn.textContent = playing ? 'Pause' : 'Play';
}

let last = performance.now();
let urlT = -1;
// Arrow keys play one segment: from a cue up to the next cue, where the clock stops and waits.
let stopAt = null;
function loop(now) {
  // The first frame timestamp can precede the time this script started, so never step back.
  const dt = Math.max(0, (now - last) / 1000); last = now;
  if (playing) {
    elapsed += dt;
    let next = t + dt;
    const g = gate();
    holding = !g.done && next >= g.end - 0.001;
    if (holding) next = g.end - 0.001;
    // Stop just short of the next cue so its content has not started; the next press begins it.
    // While a clip is mid-play the stop moves on to the following cue, so the clock always halts on a cue.
    if (stopAt !== null && next >= stopAt - 0.05) {
      if (midClip()) { const n = CUES.find(c => c > stopAt + 0.01); stopAt = n !== undefined ? n : null; }
      else { next = stopAt - 0.05; playing = false; stopAt = null; }
    }
    t = Math.min(TOTAL, next);
    if (t >= TOTAL) playing = false;
  } else {
    holding = false;
  }
  render();
  // Keep the clock in the URL so a reload returns to the same moment.
  if (Math.abs(t - urlT) >= 1 || (t === 0 && urlT !== 0)) {
    urlT = t;
    const url = new URL(location.href);
    url.searchParams.set('t', t.toFixed(1));
    history.replaceState(null, '', url);
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// Controls.
playBtn.addEventListener('click', () => { playing = !playing; });
scrub.addEventListener('input', () => { t = parseFloat(scrub.value); if (t === 0) elapsed = 0; });
themeSel.addEventListener('change', () => setTheme(themeSel.value));
capsBox.addEventListener('change', () => setCaptions(capsBox.checked));
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); playing = !playing; stopAt = null; }
  if (e.key === 't') setTheme(THEMES[(THEMES.indexOf(document.body.dataset.theme) + 1) % THEMES.length]);
  if (e.key === 'c') setCaptions(document.body.classList.contains('nocaps'));
  const list = e.shiftKey ? SCENES.map(s => s.start) : CUES;
  const playSegment = from => {
    const end = list.find(c => c > from + 0.01);
    t = from; playing = true; stopAt = end !== undefined ? end : null;
  };
  if (e.code === 'ArrowRight') {
    // Waiting exactly on a cue plays that cue's segment; anywhere else jumps to the next cue first.
    const onCue = !playing && list.some(c => Math.abs(c - t) < 0.02);
    const n = onCue ? t : list.find(c => c > t + 0.01);
    if (n !== undefined) playSegment(n); else { t = TOTAL; playing = false; }
  }
  if (e.code === 'ArrowLeft') { const p = [...list].reverse().find(c => c < t - 0.3); playSegment(p !== undefined ? p : 0); }
});

// Read-only state for tests.
window.__state = () => ({ t, elapsed, holding, playing, scene: activeScene().id, theme: document.body.dataset.theme });

// Pause when the parent website hides the film or another player takes over.
window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== window.parent || event.data?.type !== 'agentamp-film-pause') return;
  playing = false;
  document.querySelectorAll('video').forEach(video => video.pause());
  playBtn.textContent = 'Play';
});
