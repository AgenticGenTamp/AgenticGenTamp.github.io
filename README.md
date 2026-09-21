# AgenticGenTAMP project website

The formal project website for **Coding Agents for Generalized Task and Motion Planning Problems**. This is a static GitHub Pages site with no build step or runtime dependencies.

**Deployment repository:** https://github.com/AgenticGenTamp/AgenticGenTamp.github.io

**Website:** https://AgenticGenTamp.github.io/

## Run locally

From this directory:

```sh
python3 serve.py 8770
```

Open http://127.0.0.1:8770/. The included server supports byte ranges for seeking in videos. Use HTTP instead of opening `index.html` as a file: the page fetches JSON and loads JavaScript modules.

## Page structure

- Header and title: the PRPL logo, the full paper title, authors and numbered affiliations, a background rollout mosaic, and matching Paper, Code, Leaderboard, and Gallery links. Author names link to their personal homepages; the asterisk identifies the corresponding author.
- Setup: a paper-grounded definition of generalized TAMP (Introduction and Section II), three instances of the same Obstruction task, and the synthesis/evaluation protocol with links to Sections III-A and IV-A.
- Gallery: examples near the top of the page. Clips loop silently when visible, with native playback controls. Captions describe the behavior and identify the coding agent and the paper's **Main setting / + source** terminology. Seeds and episode identifiers remain in the data, but are not displayed on cards.
- Project video: a normal MP4 with native controls and a fixed 16:9 frame.
- Leaderboard: all six methods from the paper, with environment-family and matched-planner-coverage controls. Click the score heading to reverse sorting. An unranked “Future coding agents” placeholder appears above the measured methods with an unknown score; it is excluded from benchmark data and sorting.
- Environment explorer: all 28 environments, their verbatim agent-facing descriptions from archived main-setting runs, all six method rows, and printed run-level ranges. Descriptions include the original task, observation, action, and reward/goal text. The panel is visible by default and scrolls for long tables; a link opens the unchanged Markdown file.

The background and gallery videos pause off screen or when the tab is hidden. Reduced-motion and data-saving preferences disable their automatic playback. Manually paused gallery clips stay paused when returning to them. The main video plays only on request and pauses other visible gallery clips while playing.

## Sources and result definitions

`assets/paper.pdf` is the manuscript supplied on 19 September 2026. Its SHA-256 is recorded in `data/benchmark.json`. It remains the original manuscript, including its existing links. The website's author list was supplied separately by the project maintainer, who also confirmed the affiliation mapping and Tom Silver's corresponding-author designation. No arXiv identifier has been added.

The author order is Matteo Merler, Bowen Li, Josh Roy, Yichao Liang, Qianwei Wang, Yixuan Huang, and Tom Silver. The confirmed affiliations are Fondazione Bruno Kessler for Matteo, Carnegie Mellon University for Bowen, University of Cambridge for Yichao, and Princeton University for Josh, Qianwei, Yixuan, and Tom. Institutional names and homepage links were checked against the [PRPL team page](https://prpl-group.com/#team) and the personal homepages of [Matteo](https://merlerm.github.io/), [Bowen](https://jaraxxus-me.github.io/), and [Yichao](https://yichao-liang.github.io/).

All success rates and run-level min/max values are transcribed from **Tables I–II**, checked against the rendered table page. Equal-environment averages use these printed two-decimal entries, so aggregates can differ slightly from averages of unrounded source logs. A zero is a measured failure rate; `null` means no planner was provided.

- All-environment scope: 28 environments; the planner covers 16 and is shown without a rank.
- Planner scope: exactly the same 16 environments for every method.
- Source-access reference: always shown separately, unranked, because its interface differs from the main setting.
- Min/max in the explorer are across five runs, not confidence intervals.
- Table III computation times use 44 matched seeds across 13 environments where both settings reach 100% success. They are not all-environment averages.

### Media and branding

`film/` is vendored from https://github.com/merlerm/agentamp-video at commit `d81f2c1`. The original HTML, CSS, JavaScript, local fonts, stills, and required clips are retained as presentation source material. Local-path metadata JSON and development scripts are excluded. The page no longer embeds the HTML player.

`assets/project-video.mp4` is an export of that presentation in recording mode, with the default Shelf example and captions disabled. The presentation's original on-screen explanations, speed labels, and waits for unfinished clips are preserved. It is a silent, 1280 × 720, 25 fps H.264 video, encoded with CRF 23, YUV 4:2:0, and MP4 fast-start. Its duration is 3 minutes 26.2 seconds. The recording was checked for media-loading errors and sampled across all scenes. The poster is a frame from this export.

To export a future revision, serve `film/` over HTTP, record its 1280 × 720 stage using `?record=1&theme=poster`, and let its clock finish, including clip-completion holds. Use the recording helpers in the upstream presentation repository as a starting point. Hide the authoring controls and use the default on-screen explanations; the optional narration captions are a draft. Convert the recording to H.264 MP4 with `-pix_fmt yuv420p -movflags +faststart`, and keep the file under GitHub's 100 MiB per-file limit. Replace both the MP4 and its poster, then check playback and seeking.

The hero and gallery posters are derived from the supplied clips. Setup stills show Obstruction3D run 24, episodes 26, 92, and 84 (one, two, and four obstacles). Gallery provenance is recorded in `data/gallery.json`; media names retain the upstream run and episode identifiers.

`assets/prpl-robot.png` is the unmodified PRPL robot logo from https://prpl-group.com/assets/images/prpl-robot.png. It links to the Princeton Robot Planning and Learning lab website.

## Update content

### Paper and links

Replace `assets/paper.pdf`, update the SHA-256 and version in `data/benchmark.json`, and refresh table data if the manuscript changes. Paper links are local. Research-code links point to https://github.com/tomsilver/robocode; edit the links in `index.html` if the team moves it. Website-source links point to this formal repository.

### Leaderboard

Edit `data/benchmark.json`. Every environment must include one result for each method ID, using `{ "mean": 0.74, "min": 0.50, "max": 0.99 }` or `null`. Run `python3 scripts/export_csv.py` so the downloadable table matches the interactive one. Keep the evaluation protocol and source version explicit. Do not mix newer partial reruns into a complete published table without identifying their coverage.

### Environment descriptions

`data/environment-descriptions/` contains 28 byte-for-byte copies of the original `env_description.md` saved during the experiments. These are the **main-setting environment descriptions**, not summaries or reconstructions from current simulator code. They are the environment-description portion of the initial prompt, rather than the complete system/task prompt. Preserve the text even when it contains historical wording or imperfect metadata; changing it would no longer show the exact input.

`sources.json` records each website environment's experiment key, environment implementation, archive and member names, number of identical archived copies, and SHA-256. All 28 descriptions were checked against at least five archived runs. The four KinDER family archives were matched by MD5 against the final experiment collection. Packing and Blocked come from the final PDDLStream archive; Rovers comes from the strict main-setting Rovers archive; BaseMotion comes from the final Codex archive. No private Drive links, local filesystem paths, or full agent session logs are published.

The experiment code establishes the input path: `experiments/run_experiment.py` writes `env.env_description_blackbox` in the main setting; `src/robocode/approaches/agentic_base.py` reads that file and passes it to `build_agentic_prompt` in `src/robocode/prompts.py`. This was checked against local robocode commit `e437d7ab01f6815bab606ffe0ba56362c470677e`; the archived bytes, rather than regenerated output from that checkout, are the source of truth.

Mapping matters: `Shelf3D` in this site is the **dynamic** `dynamicshelf3d_generalized` environment, not the distinct kinematic `shelf3d_generalized` environment. `BaseMotion3D` uses `kinder/BaseMotion3D-v0`. The three PDDLStream tasks use `pr2packed_generalized`, `pr2blocked_generalized`, and `rovers_generalized`.

`data/environment-descriptions.json` contains the display HTML generated from the original Markdown. To regenerate it after replacing a description with a verified archival version and updating its provenance:

```sh
# Required only for regenerating this static content, not for serving the site.
python3 -m pip install markdown-it-py==3.0.0
python3 scripts/export_env_descriptions.py
npm run check
```

The renderer disables raw HTML and changes only heading levels for nesting in the page. It does not rewrite the source text. The regular checks verify all 28 mappings and original-file checksums, including the dynamic Shelf and BaseMotion distinction.

### Paper-grounded copy

Keep the definition of generalized TAMP broader than program synthesis: the Introduction describes reusable solutions such as samplers, feasibility predictors, search heuristics, and abstractions. **AgenticGenPlan** is the coding-agent synthesis approach investigated in this paper. Section II defines the shared MDP and initial-state distribution, fully observed object-centric states, frozen policies, and evaluation metrics. Sections III-A and IV-A support the prompt contents, simulator interface, sandbox/source-access distinction, $20 model-usage budget, five runs, 100 shared held-out instances, and 60-second timeout. Do not imply that every environment varies its object count or that the website's short explanation is a verbatim prompt.

### Gallery

Add an MP4 to `film/assets/clips/`, a JPEG poster to `assets/posters/`, and an entry to `data/gallery.json`. The `file` value is the common basename without the extension. Include `id`, `title`, `description`, `category`, `environment`, `method`, `backend`, `setting`, `seed`, and `episode`. Use `Main setting` or `+ source` for `setting`. Supported categories are `Strategy`, `Tool use`, and `Recovery`. Clips play at 1× in the gallery; the project video uses its own labeled playback speeds.

Keep titles descriptive and explain the observed behavior in plain language. Seeds, episode numbers, and method provenance should remain in the data even though the page shows a shorter caption.

The formal site is independent of the earlier anonymous gallery's Drive sync process. It does not change or invoke that pipeline.

## Verify

With Node.js and Python 3 available:

```sh
npm test
npm run check
```

The checks cover comparison scopes, missing-vs-zero data, aggregate ranking, all filter combinations, run ranges, all 28 original description checksums and environment mappings, local assets, the paper checksum, and accidental private paths. Review desktop/mobile layout and video playback in a browser before publishing visual changes.

## Deploy

GitHub Pages serves the `main` branch at the repository root (`/`). `.nojekyll` disables Jekyll processing. No workflow or build service is needed.

Before pushing, verify the destination:

```sh
git remote get-url origin
# Must be https://github.com/AgenticGenTamp/AgenticGenTamp.github.io.git
git push origin main
```

Never run the anonymous site's publication scripts from this repository.
