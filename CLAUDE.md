# CLAUDE.md — game-agent

Working conventions for this repo. These adapt Bastien's general dev rules (from
his other projects) to a **Python reinforcement-learning research codebase** —
so they favour clarity and small experiments over the heavy DDD/hexagonal
machinery that suits a domain app but would be accidental complexity here.

## What this project is

A step toward **one generic game-playing agent** (see `ROADMAP.md`). Today: a
curiosity-driven RL agent (PPO + RND) on Super Mario Bros, behind a game-agnostic
`GameEnv` interface. Stack: Python 3.9, PyTorch, Stable-Baselines3, Gymnasium,
`gym-super-mario-bros`. Everything runs in Docker (CPU-only on this Mac).

## Architecture (the boundary that matters)

Keep the **agent** (the brain) separate from the **game** (the adapter):
- `src/game_env.py` — the generic interface: `GameSpec` + shared obs pipeline +
  `make_venv`. **Never mention a specific game here.**
- `src/games/<name>.py` — one adapter per game (raw env + `progress_key`/
  `success_key`), registered in `src/games/__init__.py`.
- Training/eval/record scripts take `--game` and stay game-agnostic.

**Adding a game = one new file in `src/games/`.** If a change forces you to edit
the training code to support a game, the boundary leaked — stop and reconsider.

**Guiding principle — no human indication**: the agent learns from raw pixels
only. No hand-crafted perception (template matching, scripted state) and no
per-game reward engineering. (This is why the old OpenCV `detection.ipynb` was
removed.)

## Running things (Docker only)

Nothing is installed on the host. The container `mario-jupyter` is usually up
(`src/` and `data/` are mounted live, so edits apply without rebuild):
```
docker exec mario-jupyter bash -c "cd /app && python src/train_ppo.py --game mario --timesteps 100000 --n-envs 8"
docker exec mario-jupyter bash -c "cd /app && python src/eval_models.py --game mario"
```
Compose lives in `docker/`. See `README.md` for the VSCode tasks.

## Performance (CPU, benchmarked on this box)

Defaults in the training scripts are tuned for this machine (16 logical /
8 physical cores, Mario). End-to-end PPO throughput, measured:

- **`n_epochs=4`** (was 10): ~190 → ~353 steps/s (**+86%**). Biggest lever —
  the update loop was dominated by redundant gradient passes. It's a *learning*
  knob too, so re-validate the curve if you raise it. `--n-epochs` to override.
- **`n_envs=8`**: throughput plateaus past 8 (4→290, 8→353, 12→363, 16→374) —
  past 8 it's <3%/step for 2× the RAM. Collection is the floor: `nes-py` is
  single-threaded C, so more envs ≈ more emulators, not faster ones.
- **`torch` threads = physical cores** (via `src/perf.py`, imported *first*):
  torch at all 16 logical CPUs oversubscribes the 8 physical → ~12% slower than
  8. `perf.py` also caps env-worker BLAS to 1 thread. `--torch-threads` to tune.
- **RND** trains its predictor once per rollout on minibatches (`RNDTrainCallback`),
  not per env step — keeps the backward pass off the acting hot path.
- Tried and **dropped**: `channels_last` memory format (no measurable gain —
  SB3 feeds contiguous NCHW and oneDNN already handles it).

**No GPU on this Mac, and it's not a config gap.** Docker Desktop runs a Linux
VM via Hypervisor.framework, which exposes no virtual GPU; Metal/MPS has no
container passthrough (confirmed 2026). `torch.backends.mps.is_available()` is
`False` inside the container. The only GPU path is running *natively* on the
host (breaks the Docker-only convention) — treat it as a deliberate, separate
decision, not a quick win.

## ⚠️ Never clobber trained models

Training scripts overwrite `data/models/<game>_<variant>_final.zip` by default,
and `data/models/` is git-ignored (no recovery). A ~1M-step model is ~70 min of
compute. **When smoke-testing, never write to a real model's path** — use a
throwaway path/`--out`, or a scratch `--game`, or delete the tiny test artifact
after. Look before overwriting anything in `data/`.

## Code style

- **Comments = WHY only.** Names, types and structure say the *what*. Keep a
  comment only if it carries a *why* not derivable from the code (a non-obvious
  workaround, an invariant, a surprising choice). Delete paraphrases and
  temporal/planning glosses ("now", "Phase 2", "TODO later").
- **Guard clauses over `else`.** Prefer early returns; extract a function rather
  than nesting. Short ternaries are fine.
- **No primitive obsession where it earns its keep**, but don't wrap everything —
  this is research code, keep it light.
- **English** for all code, identifiers, comments and docs. **French** for
  conversation with Bastien (with the English-coaching side-channel).

## Tests (when we add them)

RL is hard to unit-test, so there are none yet. When adding tests:
- Name: `test_should_<observable_outcome>_when_<condition>()` — `should_`/`when_`
  are non-optional; describe the outcome, not the mechanism.
- Body: explicit `# given` / `# when` / `# then` blocks.
- **DAMP over DRY**: prefer readable inline duplication over helpers that make
  you scroll. Local `given_*/when_*/then_*` helpers are fine.

## Git & commits

- **Conventional Commits**: `type(scope): summary` (`feat`, `fix`, `refactor`,
  `docs`, `chore`, `perf`, `test`…).
- **User-oriented description** — say what changes for the user/experiment, not
  the low-level mechanics (the summary should read well in a changelog).
- End every commit message with:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```
- Bastien runs commits/pushes himself — propose the message, don't run `git
  commit`.

## When to ask vs. act

- Refactor freely *within* a module/adapter.
- **Defer cross-cutting changes** (touching the generic interface + every
  adapter/script at once, big dependency swaps) — propose them, let Bastien
  decide.
- Report outcomes honestly: if a run fails or a model was overwritten, say so
  plainly with the evidence.
