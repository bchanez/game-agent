"""LLM reasoning agent for ARC-AGI-3 — the "understand the puzzle, then act" path.

Pure RL (PPO+SIL+Go-Explore) cracks ls20 level 1 but not level 2: the level-2
mechanic (a required path under a depleting life budget) is a *discovery* wall that
trial-and-error doesn't cross (see FINDINGS). This agent tests the other axis —
**reasoning**: an LLM reads the grid *as text* (ARC grids are symbolic, so no vision
needed), reasons about what each action does and what the goal is, acts, observes the
result, updates its notes, and repeats. This is Bastien's brain-loop — perceive →
reason → act → observe → remember — with a capable model as the reasoner.

⚠️ Research/dev prototype only. ARC-AGI-3's competition eval runs offline on Kaggle
with no external LLM/API (see docs/arc-agi-3.md), so this cannot be the *submission* —
it tests whether reasoning breaks the discovery wall the RL stack couldn't.

    python src/arc_llm_agent.py --game arc_ls20 --dry-run          # no API, validates the loop
    python src/arc_llm_agent.py --game arc_ls20 --max-steps 60     # real run (needs ANTHROPIC_API_KEY)

Needs `pip install anthropic` and ANTHROPIC_API_KEY for a real run.
"""
import perf  # first: BLAS thread limits before numpy

import argparse
import json
import random

import numpy as np

from games import get_game

SYMBOLS = "0123456789abcdef"                 # 16 colors -> one char each
ACTION_NAMES = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]

SYSTEM = """You are an agent dropped into an unknown grid-puzzle game (ARC-AGI-3).

You see a 64x64 grid; each cell is one of 16 colors, shown as a character 0-9a-f.
You can take these actions, whose meaning is UNKNOWN and VARIES per game — you must
discover what each does by acting and observing:
  ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, and ACTION7 (usually "undo").
There is NO stated goal and NO rules. Infer the goal from what changes. The only
feedback is `levels_completed`, which increases when you make real progress — treat
raising it as the objective, and avoid states that end the game.

Play like a scientist: form a hypothesis about an action, test it, read the result,
and keep a running set of notes about what you've learned (action meanings, the goal,
the mechanic, mistakes to avoid). Think before each move; then commit to ONE action."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "brief: what you infer and why this action"},
        "notes": {"type": "string", "description": "your updated running notes (carried to next turn)"},
        "action": {"type": "string", "enum": ACTION_NAMES},
    },
    "required": ["reasoning", "notes", "action"],
    "additionalProperties": False,
}


def grid_to_text(grid):
    return "\n".join("".join(SYMBOLS[int(c)] for c in row) for row in grid)


def diff_summary(prev, cur):
    if prev is None or prev.shape != cur.shape:
        return "n/a (first frame)"
    changed = int(np.count_nonzero(prev != cur))
    if changed == 0:
        return "0 cells changed (no visible effect)"
    ys, xs = np.nonzero(prev != cur)
    return (f"{changed} cells changed, in rows {ys.min()}-{ys.max()}, "
            f"cols {xs.min()}-{xs.max()}")


def choose_action_llm(client, model, grid_text, notes, history_text):
    user = (f"Your notes so far:\n{notes or '(none yet)'}\n\n"
            f"Recent moves (action -> effect, reward):\n{history_text or '(none yet)'}\n\n"
            f"Current 64x64 grid:\n{grid_text}\n\n"
            "Reason, update your notes, and choose the next action.")
    resp = client.messages.create(
        model=model, max_tokens=3000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    return data["action"], data.get("reasoning", ""), data.get("notes", notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="arc_ls20")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true",
                    help="random actions, no API call — validates the loop for free")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    client = None
    if not args.dry_run:
        import anthropic                       # lazy: dry-run needs no SDK
        client = anthropic.Anthropic()

    env = get_game(args.game).make_raw_env()
    obs, info = env.reset(seed=args.seed)
    grid = np.asarray(obs)[0]
    notes, history, prev = "", [], None
    levels, reasons = 0, []

    for step in range(args.max_steps):
        grid_text = grid_to_text(grid)
        hist_text = "\n".join(history[-8:])
        if args.dry_run:
            name = random.choice(ACTION_NAMES)
            reasoning = "(dry-run: random)"
        else:
            name, reasoning, notes = choose_action_llm(
                client, args.model, grid_text, notes, hist_text)

        obs, reward, term, trunc, info = env.step(ACTION_NAMES.index(name))
        cur = np.asarray(obs)[0]
        effect = diff_summary(grid, cur)
        levels = info.get("levels_completed", levels)
        history.append(f"{name} -> {effect}, reward={reward:+.0f}, levels={levels}")
        reasons.append(f"[{step}] {name}: {reasoning}")
        print(f"[{step:3d}] {name:8} {effect:45} reward={reward:+.0f} levels={levels}", flush=True)
        prev, grid = grid, cur
        if term or trunc:
            print(f"--- episode ended (won={info.get('won')}) at step {step} ---", flush=True)
            break

    print(f"\nDone: {step+1} steps, deepest levels_completed = {levels}", flush=True)
    if not args.dry_run:
        print("\nFinal notes:\n" + notes, flush=True)


if __name__ == "__main__":
    main()
