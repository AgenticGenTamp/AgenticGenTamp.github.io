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

- Header and title: the PRPL logo, the full paper title, authors and numbered affiliations, a background rollout mosaic, and matching Paper, arXiv, Code, Experimental results, and Gallery links. Author names link to their personal homepages.
- Setup: a paper-grounded definition of generalized TAMP (Introduction and Section II), three instances of the same Obstruction task, and the synthesis/evaluation protocol with links to Sections III-A and IV-A.
- Gallery: 36 examples near the top of the page, with four compact cards per row on desktop, two on tablets, and one on small screens. Examples are split into two groups, "Strategies and behaviors" and "Failures" (the failure cards plus the two successes of the ConstrainedCupboard story), each in its own scrollable box about two rows tall so the page stays short. Cards that tell one story share an optional `story` field; consecutive cards with the same story render together in a labeled, tinted panel (the ConstrainedCupboard trio: three rods on the lower shelves, the fourth rod failing, and + source using the higher shelves). Within each group, the 3D clips come first: cards follow the environment families in the order Dynamic3D, Kinematic3D, PDDLStream, Dynamic2D, Kinematic2D, and cards for the same environment sit together. Clips loop silently when visible at 8× by default, with a 1×/2×/4×/8× selector and native playback controls. Captions describe the behavior and identify the coding agent and the paper's **Main setting / + source** terminology. Seeds and episode identifiers remain in the data, but are not displayed on cards.
- Project video: a normal MP4 with native controls and a fixed 16:9 frame.
- Experimental results: all six methods from the paper plus Codex with GPT-6 Astra in the main setting and + source, each from 140 complete archived runs, with environment-family and matched-planner-coverage controls. The table shows results without a rank column or winner highlight. Click the score heading to reverse sorting. A “Future coding agents” placeholder appears above the measured methods with an unknown score; it is excluded from benchmark data and sorting.
- Environment explorer: all 28 environments, with synthesized policy videos first, all eight method results and run-level ranges second, and the verbatim agent-facing description last. Selecting an environment starts its available method clips together when the panel is visible. The separate illustration video is no longer displayed. Descriptions include the original task, observation, action, and reward/goal text from archived main-setting runs. The input section states that the agent receives text and interactive simulator access, without demonstration videos. It scrolls for long tables; a link opens the unchanged Markdown file.

The background and gallery videos pause off screen or when the tab is hidden. Reduced-motion and data-saving preferences disable their automatic playback. Manually paused gallery clips stay paused when returning to them. The main video plays only on request and pauses other visible gallery clips while playing.

## Sources and result definitions

`assets/paper.pdf` is arXiv:2609.30233v1 (https://arxiv.org/abs/2609.30233), released 25 September 2026. Its SHA-256 is recorded in `data/benchmark.json`. The website's author list was supplied separately by the project maintainer, who also confirmed the affiliation mapping.

The author order is Matteo Merler, Bowen Li, Josh Roy, Yichao Liang, Qianwei Wang, Yixuan Huang, and Tom Silver. The confirmed affiliations are Fondazione Bruno Kessler for Matteo, Carnegie Mellon University for Bowen, University of Cambridge for Yichao, and Princeton University for Josh, Qianwei, Yixuan, and Tom. Institutional names and homepage links were checked against the [PRPL team page](https://prpl-group.com/#team) and the personal homepages of [Matteo](https://merlerm.github.io/), [Bowen](https://jaraxxus-me.github.io/), and [Yichao](https://yichao-liang.github.io/).

The original six methods’ success rates and run-level min/max values are transcribed from **Tables I–II**, checked against the rendered table page. The additional **Codex with GPT-6 Astra** results are computed from 140 complete archived runs in `Final/Blackbox/Codex GPT6 Astra`, five synthesis seeds and 100 held-out episodes per run. The `bad - too little tokens` folder is excluded, and duplicate revisions are resolved by selecting the latest complete run for each environment/seed. `data/astra-results.json` records all 140 run identities, result and program hashes, success counts, and unrounded means. Astra’s environment means are rounded to two decimals (round half up), matching the paper’s displayed precision. Equal-environment averages use these rounded means, so aggregates can differ slightly from averages of unrounded source logs. A zero is a measured failure rate; `null` means no planner was provided.

The website totals include the paper’s 700 programs and 70,000 evaluation episodes plus Astra’s 280 programs and 28,000 episodes (140 programs in each setting): 980 programs and 98,000 episodes. `data/benchmark.json` records each part under `protocol`. The PDF itself is unchanged.

### Astra + source

`data/astra-source-results.json` holds **AgenticGenPlan + source with Codex and GPT-6 Astra (high)** on all 28 environments, from 140 complete archived runs in `Final/Whitebox/GPT-6 Astra`: gpt-6-astra with high reasoning effort, $20 budget, source access, synthesis seeds 24/42/222/424/444, 100 held-out episodes per run, and evaluation seed 792075 as in the main-setting runs. All 14,000 episodes completed without crashes. Runs inside multi-run bundles are included, and the latest complete run is selected for each environment and seed. Per-run rates are counted from the 100 per-episode outcomes in each run's `results.json`; means use the same two-decimal rounding as Astra's main-setting results. The 28 rounded means and min/max values agree exactly with the Astra + source row of the updated manuscript's Tables I–II. The file records archive and member names, experiment IDs, result and program hashes, success counts, and unrounded means.

Like Claude Code with Opus 5 + source, this row is a + source reference: it appears in the results table without a rank, in the environment explorer, and in the “Effect of source access” comparison, which averages each agent's main setting and + source over all 28 environments.

- All-environment scope: 28 environments; the planner covers 16; its partial coverage is stated explicitly.
- Planner scope: exactly the same 16 environments for every method.
- + source references: Claude Code with Opus 5 + source and Codex with GPT-6 Astra + source are always shown separately because their interface differs from the main setting.
- Min/max in the explorer are across five runs, not confidence intervals.
- Table III computation times compare Claude Code with Opus 5, Claude Code with Opus 5 + source, Codex with GPT-6 Astra, and Codex with GPT-6 Astra + source on the 15 environments where all four settings have at least one run with 100% held-out success. Each method uses its own runs with 100% held-out success; values are equal-environment means with min/max across environment-level means. The two GPT-6 Astra rows are recomputed here from their archived runs (`data/astra-source-results.json` holds the + source per-environment values) and match the manuscript.
- Table III in the manuscript and on this website covers the 15 environments where all four settings have at least one run with 100% held-out success: 11.741, 42.480, 1.299, and 50.392 ms/action.

### Media and branding

`film/` is vendored from https://github.com/merlerm/agentamp-video at commit `d81f2c1`. The original HTML, CSS, JavaScript, local fonts, stills, and required clips are retained as presentation source material. Local-path metadata JSON and development scripts are excluded. The page no longer embeds the HTML player.

`assets/project-video.mp4` is the supplied `public-2026-09-23.mp4`, kept byte-for-byte unchanged (SHA-256: `069578f63710515b7f3271c891103a4032aff69341012ae07e886f999670c096`). It is a 1280 × 720, 30 fps H.264 video with YUV 4:2:0, AAC audio, and MP4 fast-start. Its duration is 2 minutes 58.6 seconds. The complete audio and video streams decode without errors. The poster is the title frame at 12 seconds. This public-release video supersedes the previous submission video.

For future replacements, use browser-compatible H.264 MP4 with AAC audio when present, YUV 4:2:0, and fast-start, keeping the file under GitHub's 100 MiB per-file limit. Replace the MP4 and poster together and update their versioned links in `index.html` and the fallback in `app.js`. Check the full file for decoding errors. The vendored `film/` remains the source for the hero and gallery clips; replacing the project video does not update those clips.

The hero and gallery posters are derived from the supplied clips. Setup stills show Obstruction3D run 24, episodes 26, 92, and 84 (one, two, and four obstacles). Gallery provenance is recorded in `data/gallery.json`; media names retain the upstream run and episode identifiers.

`assets/prpl-robot.png` is the unmodified PRPL robot logo from https://prpl-group.com/assets/images/prpl-robot.png. It links to the Princeton Robot Planning and Learning lab website.

## Update content

### Paper and links

Replace `assets/paper.pdf`, update the SHA-256 and version in `data/benchmark.json`, and refresh table data if the manuscript changes. Paper links are local. Research-code links point to https://github.com/tomsilver/robocode; edit the links in `index.html` if the team moves it. Website-source links point to this formal repository.

### Experimental results

Edit `data/benchmark.json`. Every environment must include one result for each method ID, using `{ "mean": 0.74, "min": 0.50, "max": 0.99 }` or `null`. Run `python3 scripts/export_csv.py` so the downloadable table matches the interactive one. Keep the evaluation protocol and source version explicit. Do not mix newer partial reruns into a complete published table without identifying their coverage.

### Environment descriptions

`data/environment-descriptions/` contains 28 byte-for-byte copies of the original `env_description.md` saved during the experiments. These are the **main-setting environment descriptions**, not summaries or reconstructions from current simulator code. They are the environment-description portion of the initial prompt, rather than the complete system/task prompt. Preserve the text even when it contains historical wording or imperfect metadata; changing it would no longer show the exact input.

`sources.json` records each website environment's experiment key, environment implementation, archive and member names, number of identical archived copies, and SHA-256. All 28 descriptions were checked against at least five archived runs. The four KinDER family archives were matched by MD5 against the final experiment collection. Packing and Blocked come from the final PDDLStream archive; Rovers comes from the main-setting Rovers archive; BaseMotion comes from the final Codex archive. No private Drive links, local filesystem paths, or full agent session logs are published.

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

### Synthesized policy examples and the environment audit

`data/policy-examples.json` describes 112 verified policy clips across all 28 environments, with four methods per environment. Selecting an environment starts its available clips together from the beginning; “Replay together” restarts them. The initial page load does not start off-screen policy videos. Switching environments unloads the previous clips; scrolling away, hiding the tab, or playing the project video pauses policy playback. Astra is appended after the existing methods. Four-method comparisons use a two-by-two grid on larger screens and a single column on small screens. The explorer displays the actual clip count and makes missing methods explicit. Within each environment, available methods act on the same held-out instance. Captions identify the method, input setting, replay outcome, and action count. Videos remain separate from the agent's text input.

The JSON records exact archive and member names, program and results hashes, replicate and episode, instance seed, initial-frame hash (and an exact state hash when rasterization can differ), archived outcome, and replay action count. Initial states or frames must agree across the compared methods. All published replay outcomes match the archive; replay action counts may vary slightly from archived counts and are recorded separately. No numerical paper results are changed. Scene backgrounds are enabled for all dynamic 3D renders. All 28 Astra replay outcomes match the archives; four have different action counts (DynPushPullHook2D, DynScoopPour2D, SortClutteredBlocks3D, and SweepIntoDrawer3D). Displayed action counts describe the replay; archived counts remain in the provenance, and aggregate success rates use the original evaluations.

The original instance selection used replicate 42 from each available method and, among shared episodes with valid recorded action counts, preferred differing outcomes, then object count nearest 3, then the lowest episode index. The Codex rerun update preserved those held-out instances; the subsequent ScoopPour update described below moves all four policies together to episode 0. It prefers synthesis replicate 42 and otherwise the first available replicate in 24, 222, 424, 444 order; synthesis seeds can differ across methods, but the held-out instance and initial state must match. Astra uses synthesis replicate 42 from the same selected archives as the success-rate data, retaining each existing held-out episode except the joint ScoopPour update described below. These examples illustrate differences and are **not representative averages**. Playback samples every three actions at 10 fps, with a final one-second hold; video length is not policy computation time.

The eight previously missing archives were uploaded to Drive on 2026-09-23: Claude Code on BaseMotion, and LLMGenPlan on BaseMotion, ConstrainedCupboard, Dynamo, ScoopPour, SortClutteredBlocks, SweepSimple, and Rovers. These programs were rendered with scene backgrounds enabled. The supplied ScoopPour archive uses evaluation object counts of 10 and 100, while the other methods use 10, 20, 30, 40, and 50. Its four clips were therefore rerendered on the first episode whose count agrees across all archives (episode 0, 10 objects). All other shared held-out instances are retained. Their initial-state or initial-frame hashes agree with the other methods, and all eleven new or replaced replay outcomes and action counts match their archived episodes. No policy clips are currently missing.

Codex clips for 13 environments now come from the project maintainer-confirmed no-internet **Codex Reruns** collection. `codexRerunEnvironments` lists them, and the checks reject older collection names for those clips. The remaining 15 Codex environments retain their existing results as confirmed by the maintainer. The rerun archives contain frozen programs rather than ready-made videos, so their clips were rendered with scene backgrounds enabled. In particular, the corrected ConstrainedCupboard replay fails as recorded, and the Shelf rerun succeeds in 940 actions, allowing its previously withheld clip to be restored. Video outcomes are checked against their own archived episodes; aggregate values still come separately from the paper and are not recomputed from these selected videos.

`data/environment-audit.json` records the earlier audit of all 28 reader-only illustration panels: the original six methods’ values match Tables I–II, descriptions match archived checksums, each illustration decodes fully, and first/middle/final frames were inspected against the task. Those panels have been removed from the explorer; the historical audit and upstream assets remain for provenance. The policy clips displayed in the explorer have separate provenance in `data/policy-examples.json`. Upstream illustration files are not treated as evidence for a particular method's performance because their exact original policy/run attribution has not been independently established.

When adding policy examples, use the archived frozen program and environment configuration, enable scene backgrounds wherever supported, check the shared initial state, and verify outcomes against the experiment record. Record provenance and hashes and do not use a different or easier instance to make a method appear stronger.

### Paper-grounded copy

Keep the definition of generalized TAMP broader than program synthesis: the Introduction describes reusable solutions such as samplers, feasibility predictors, search heuristics, and abstractions. **AgenticGenPlan** is the coding-agent synthesis approach investigated in this paper. Section II defines the shared MDP and initial-state distribution, fully observed object-centric states, frozen policies, and evaluation metrics. The Experimental setup section makes the main-setting restrictions explicit: an isolated Docker container with a separate filesystem and no network or host-filesystem access, no environment implementation inside the container, only Python/NumPy/SciPy, and no supplied hand-written TAMP predicates, operators, samplers, or skills. Section III-A (pages 2–4) describes these restrictions, the client/server simulator interface, and red-team attempts to test isolation. It also describes the separate Claude Code + source setting, including importing helpers and setting arbitrary states. Sections II-C and IV-A support the frozen policy with no test-time agent/LLM calls, $20 model-usage budget per synthesis run, five runs, 100 shared held-out instances, evaluation-seed overlap checks, and 60-second timeout. Do not turn the isolation checks into a claim of a formal security guarantee, imply that every environment varies its object count, or present the website's summary as a verbatim prompt.

### Gallery

Add an MP4 to `film/assets/clips/`, a JPEG poster to `assets/posters/`, and an entry to `data/gallery.json`. The `file` value is the common basename without the extension. Include `id`, `title`, `description`, `environment`, `method`, `backend`, `setting`, `seed`, and `episode`. The `category` field (Strategy, Tool use, Recovery, or Failure) is not displayed; it places Failure cards in the "Failures" group and all others in "Strategies and behaviors". The optional `group` field (`"strategies"` or `"failures"`) overrides that placement without changing the category, so a success can sit next to the failures it explains. The ConstrainedCupboard story uses it: a three-rod success, the same program failing with four rods, and a + source program solving that four-rod instance. The page keeps the order of `data/gallery.json`. Use `Main setting` or `+ source` for `setting`. Clips play at 8× by default, with selectable 1×, 2×, 4×, and 8× playback. The optional `holdEnd` field (seconds, at most 10) keeps a clip's last frame on screen for that much real time at every playback speed before the clip restarts; such clips do not use native looping, and the hold follows the same visibility, reduced-motion, and user-pause rules as other clips. The Packing square example uses it to show its final top-down view for 2 seconds. The project video uses its own labeled playback speeds.

Keep titles descriptive and explain the observed behavior in plain language. Seeds, episode numbers, and method provenance should remain in the data even though the page shows a shorter caption.

Failure cards use `category: "Failure"` and a title that starts with “Failure:”. The checks enforce that every Failure card replays an archived failure and sits in the "Failures" group, that only Failure cards have titles starting with “Failure”, that `group` is valid, and that within each group the families appear in the order above with cards for the same environment adjacent.

Gallery clips with a `source` object are replays of a frozen archived program on the recorded held-out episode. `source` records the collection, archive and member names, environment ID, synthesis seed, episode, instance reset seed, object count, result and program hashes, archived and replayed outcome and step count, initial-frame hash, clip hash, and why the episode was chosen. They follow the results viewer's two-pass replay: the policy runs once without rendering while its actions are recorded, then the recorded actions are replayed with rendering, so rendering time cannot change the policy's decisions. Scene backgrounds are on; frames are 1.5× the native render size (960 × 720, or 960 × 540 for the 16:9 Kinematic3D cameras); one frame per action at 10 fps, plus ten settle steps for dynamic environments and eight hold frames. Clips are H.264 High, YUV 4:2:0, CRF 22, fast-start, without audio; posters are 720-pixel-wide JPEG frames from the clip. Packing clips append a two-second top-down still of the final state (`appendedTopDownFrames`). Very long episodes may keep only every second or fourth action frame (`frameStride`), which the card description states. Cards with `renderPass: "single"` were rendered during the policy run itself, used only when the taped second pass did not reproduce the archived outcome; the selection note says so. Only episodes whose replay reproduces the archived outcome are published; replay step counts can differ slightly and are recorded next to the archived ones. GPT-6 Astra hashes are checked against `data/astra-results.json` or `data/astra-source-results.json`, and the clip hash against the file.

The formal site is independent of the earlier anonymous gallery's Drive sync process. It does not change or invoke that pipeline.

## Verify

With Node.js and Python 3 available:

```sh
npm test
npm run check
```

The checks cover comparison scopes, missing-vs-zero data, aggregate ranking, all filter combinations, run ranges, all 28 original description checksums and environment mappings, local assets, the paper checksum, the 140 Astra and 140 Astra + source success counts and aggregation, the Table III timing values and bar widths, and accidental private paths. Review desktop/mobile layout and video playback in a browser before publishing visual changes.

## Deploy

GitHub Pages serves the `main` branch at the repository root (`/`). `.nojekyll` disables Jekyll processing. No workflow or build service is needed.

Before pushing, verify the destination:

```sh
git remote get-url origin
# Must be https://github.com/AgenticGenTamp/AgenticGenTamp.github.io.git
git push origin main
```

Never run the anonymous site's publication scripts from this repository.
