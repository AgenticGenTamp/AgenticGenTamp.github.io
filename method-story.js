// A visible, linear story. The videos themselves drive stage transitions.
export function mountMethodStory(section, {reduced, connection, pauseOtherMedia = () => {}}) {
  const doc = section.ownerDocument;
  const win = doc.defaultView;
  const stages = [...section.querySelectorAll('[data-method-stage]')];
  const videos = stages.map(stage => stage.querySelector('video'));
  const progress = stages.map(stage => stage.querySelector('progress'));
  const toggle = section.querySelector('#method-toggle');
  const status = section.querySelector('#method-status');
  const calibrationLabel = section.querySelector('#calibration-video-label');
  const titles = ['Test the model', 'Fit & refine', 'Freeze & test'];
  const visible = new Map();
  let active = 0;
  let wanted = !reduced.matches && !connection?.saveData;
  let finished = false;
  let blocked = false;
  let request = 0;
  toggle.hidden = false;

  function pause() {
    request++;
    videos.forEach(video => video.pause());
  }
  function update() {
    stages.forEach((stage, index) => {
      stage.classList.toggle('active', index === active && !finished);
      if (index === active && !finished) stage.setAttribute('aria-current', 'step');
      else stage.removeAttribute('aria-current');
    });
    section.dataset.stage = finished ? 'complete' : String(active);
    toggle.textContent = finished ? '↻ Replay walkthrough' : wanted && !blocked ? 'Ⅱ Pause walkthrough' : '▶ Play walkthrough';
    toggle.setAttribute('aria-label', finished ? 'Replay walkthrough' : wanted && !blocked ? 'Pause walkthrough' : 'Play walkthrough');
    status.textContent = finished ? 'Probe → calibrate → evaluate' : `${String(active + 1).padStart(2,'0')} / 03 · ${titles[active]}`;
  }
  function load(index) {
    const video = videos[index];
    if (!video.getAttribute('src')) {
      video.src = video.dataset.src;
      video.load();
    }
  }
  function play() {
    if (!wanted || finished || blocked || doc.hidden || !visible.has(active)) return;
    load(active);
    const video = videos[active];
    if (!video.paused) return;
    pauseOtherMedia();
    const token = ++request;
    video.play().then(() => {
      if (!wanted || doc.hidden || !visible.has(videos.indexOf(video)) || videos[active] !== video) video.pause();
    }).catch(() => {
      if (token !== request) return;
      blocked = true;
      update();
    });
  }
  function activate(index) {
    if (active !== index) pause();
    active = index;
    update();
    play();
  }
  function resumeVisible() {
    if (!wanted || finished || doc.hidden) {pause();return;}
    if (!visible.has(active)) {
      const next = [...visible.keys()].sort((a,b) => a-b).find(index => index >= active);
      if (next === undefined) {pause();return;}
      activate(next);
    } else play();
  }
  const observer = new win.IntersectionObserver(entries => {
    entries.forEach(({target, isIntersecting, intersectionRatio}) => {
      const index = videos.indexOf(target);
      if (isIntersecting && intersectionRatio >= 0.25) visible.set(index, intersectionRatio);
      else {visible.delete(index);target.pause();}
    });
    resumeVisible();
  }, {threshold:[0,0.25,0.75]});
  videos.forEach((video,index) => {
    video.muted = true;
    video.addEventListener('timeupdate', () => {
      if (Number.isFinite(video.duration) && video.duration > 0) progress[index].value = Math.min(1, video.currentTime / video.duration);
      if (index === 1) calibrationLabel.textContent = video.currentTime >= 5.55 ? 'Orange: prior prediction · gap enlarged 5×' : 'Collect block positions · 8×';
    });
    video.addEventListener('ended', () => {
      if (index !== active) return;
      progress[index].value = 1;
      stages[index].classList.add('complete');
      if (active < videos.length - 1) activate(active + 1);
      else {finished = true;pause();update();}
    });
    observer.observe(video);
  });
  toggle.addEventListener('click', () => {
    if (finished) {
      finished = false;
      active = 0;
      stages.forEach(stage => stage.classList.remove('complete'));
      videos.forEach((video,index) => {if(video.getAttribute('src')) video.currentTime = 0;progress[index].value = 0;});
      wanted = true;
    } else wanted = blocked || !wanted;
    blocked = false;
    update();
    // On mobile, play the stage currently in view; never scroll the page for the reader.
    resumeVisible();
  });
  doc.addEventListener('visibilitychange', resumeVisible);
  reduced.addEventListener('change', () => {
    wanted = !reduced.matches && !connection?.saveData;
    blocked = false;
    update();
    resumeVisible();
  });
  update();
  return {pause};
}
