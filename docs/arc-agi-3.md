# ARC-AGI-3 — target benchmark (interface + porting plan)

The long-term target for the generic agent (see `ROADMAP.md` Phase 9). ARC-AGI-3
is the **interactive** track of ARC Prize 2026: an agent is dropped into games it
has **never seen**, with **no instructions, no stated goal, no rules**, and must
figure everything out by trial and observation — exactly this project's thesis
(*one generic agent, no human indication*). Competition eval runs **offline on
Kaggle** (no internet, so **no external LLM/API** — you build and train the whole
system yourself).

This file freezes what the real interface looks like, so we design against
reality, not assumptions. Extracted from the SDK docs (Sept 2026); confirm field
names against running code when we stand up the py3.12 env (see blocker below).

## Observation

- One or more **2D frames of `64×64`**, each cell an integer **`0–15`** (16
  colors), plus game-state metadata.
- A discrete color grid — **not raw pixels**. So our Mario/Atari obs pipeline
  (frame-skip → resize 84×84 → grayscale → stack) does **not** apply; a net would
  ingest the grid directly (e.g. 16-channel one-hot `64×64`).

## Actions (7)

| Action | Meaning |
|---|---|
| `ACTION1..4` | up / down / left / right — **semantic, varies per game** |
| `ACTION5` | interact / select / execute — varies per game |
| `ACTION6` | **spatial click**, takes `x,y` coordinates in `[0,63]` |
| `ACTION7` | **undo** (revert to the previous state) |
| `RESET` | (re)initialize the game / level |

Two specifics that differ from our Mario canonical controller:
- The obs is a **discrete grid**, not pixels.
- `ACTION6` is a **spatial action** (a `(x,y)` output over the grid), not a plain
  `Discrete` button — a new output modality to design for.

## Game state & scoring

- States: `WIN`, `GAME_OVER`, `NOT_FINISHED`.
- Score via `arc.get_scorecard()`. Reward is sparse/absent → **exploration /
  curiosity is the driver** (our RND carries over conceptually).

## SDK entry points

```python
import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")   # omit render_mode for +2K FPS
env.step(GameAction.ACTION1)
arc.get_scorecard()
```

`pip install arc-agi`. `ARC_API_KEY` optional (anonymous key otherwise). Games
have string ids (e.g. `"ls20"`); a set is public, more are held out for eval.

## ⚠️ Blocker — Python version

`arc-agi` requires **Python ≥3.10** (most releases ≥3.12/3.13). The
`mario-jupyter` container is **Python 3.9**, so the SDK **cannot be installed
there**. This is a deliberate infra fork, not a quick fix:

- Steps 1–2 of the plan (meta-RL) run **entirely on Mario/Atari in the existing
  3.9 container** — they never touch ARC-AGI-3.
- The **py3.12 container + SDK** (or offline community game envs) is stood up at
  **step 3 (porting)**, when we actually need to run ARC-AGI-3 — not before.

## What carries over from this project

- **`GameEnv` boundary** (agent ⊥ game) → an ARC-AGI-3 game is a new adapter.
- **PPO + RND curiosity** → central, since ARC-AGI-3 has little/no reward.
- **SPR** (self-predictive representation) → learn dynamics without reward.
- **Meta-RL / in-context adaptation** → the paradigm for novel unseen games.

What does **not** carry: the pixel obs pipeline and the d-pad canonical action
space (ARC-AGI-3 uses a discrete grid + a spatial click action).

## Sources

- https://arcprize.org/competitions/2026
- https://docs.arcprize.org/ (quickstart, actions, game-schema)
- https://arcprize.org/arc-agi/3
